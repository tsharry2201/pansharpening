"""
渐进式蒸馏训练辅助函数
用于ARConv的渐进式激活和fixstep管理
"""
import torch

def get_training_stage(global_step, args):
    """
    根据训练步数返回当前阶段
    
    阶段说明：
    - warmup (0-100步): Student使用Conv2d（继承预训练+LoRA），无ARConv
    - activate (100-3000步): 切换到ARConv，训练所有参数（offset+modulation+weight，lr=0.1x）
    - full (3000-4000步): ARConv全力训练（lr=0.5x）
    - fixed (4000步之后): ARConv的offset/modulation固定，但权重继续训练（lr=0.5x）
    
    Args:
        global_step: 当前训练步数
        args: 训练参数
    
    Returns:
        stage: 当前阶段名称
    """
    warmup_steps = getattr(args, 'arconv_warmup_steps', 100)  # 改为100
    activate_steps = getattr(args, 'arconv_activate_steps', 3000)
    fixstep = getattr(args, 'arconv_fixstep', 4000)
    
    if global_step < warmup_steps:
        return "warmup"
    elif global_step < activate_steps:
        return "activate"
    elif global_step < fixstep:
        return "full"
    else:
        return "fixed"


def create_progressive_optimizer(model_gen, model_reg, args, stage="warmup"):
    """
    根据训练阶段创建不同的优化器
    
    Args:
        model_gen: 生成器模型
        model_reg: 正则化模型
        args: 训练参数
        stage: 训练阶段（warmup/activate/full/fixed）
    
    Returns:
        optimizer: 优化器
    """
    params_to_optimize = []
    
    # LoRA参数（始终训练）
    lora_params = []
    for n, p in model_gen.named_parameters():
        if p.requires_grad and 'lora' in n:
            lora_params.append(p)
    
    if len(lora_params) > 0:
        params_to_optimize.append({
            'params': lora_params,
            'lr': args.learning_rate,
            'name': 'lora'
        })
    
    # Scene Token参数（根据阶段调整学习率）
    scene_extractor = model_gen.module.scene_extractor if hasattr(model_gen, 'module') else model_gen.scene_extractor
    if scene_extractor is not None:
        scene_params = []
        for p in scene_extractor.parameters():
            if p.requires_grad:  # 只添加requires_grad=True的参数
                scene_params.append(p)
        
        if len(scene_params) > 0:
            # 前300步Scene Token冻结（不添加到优化器）
            # 300步后解冻，使用0.1x学习率
            scene_token_freeze_steps = getattr(args, 'scene_token_freeze_steps', 300)
            global_step = getattr(args, '_current_step', 0)  # 从args获取当前步数
            
            if global_step >= scene_token_freeze_steps:
                params_to_optimize.append({
                    'params': scene_params,
                    'lr': args.learning_rate * 0.1,  # Scene Token用0.1x学习率
                    'name': 'scene_token'
                })
                print(f"✅ Scene Token参数: {len(scene_params)} 个（lr=0.1x，已解冻）")
            else:
                print(f"❄️  Scene Token冻结中（{global_step}/{scene_token_freeze_steps}步）")
    
    # 🔥 Scene Gate参数（始终训练，正常学习率）
    unet = model_gen.module.unet if hasattr(model_gen, 'module') else model_gen.unet
    scene_gate_params = []
    for n, p in unet.named_parameters():
        if 'scene_gate' in n and p.requires_grad:
            scene_gate_params.append(p)
    
    if len(scene_gate_params) > 0:
        params_to_optimize.append({
            'params': scene_gate_params,
            'lr': args.learning_rate,  # Scene Gate用正常学习率
            'name': 'scene_gate'
        })
        print(f"✅ Scene Gate参数: {len(scene_gate_params)} 个（lr=1.0x）")
    
    # 🔥 ARConv参数（根据阶段设置学习率）
    # 注意：warmup阶段使用Conv2d，ARConv参数不存在；activate阶段后才有ARConv
    arconv_params = []
    for n, p in model_gen.named_parameters():
        # 检查是否是ARConv相关参数（包括convs, m_conv, b_conv, p_conv, l_conv, w_conv等）
        if p.requires_grad:
            # 检查参数名中是否包含ARConv特有的层
            arconv_keywords = ['m_conv', 'b_conv', 'p_conv', 'l_conv', 'w_conv', 'convs']
            if any(keyword in n for keyword in arconv_keywords):
                arconv_params.append(p)
    
    if len(arconv_params) > 0:
        if stage == "warmup":
            # Warmup阶段：使用Conv2d，ARConv参数不应该存在
            print(f"⚠️  警告: Warmup阶段检测到{len(arconv_params)}个ARConv参数，这不应该发生！")
        elif stage == "activate":
            # Activate阶段：小学习率（ARConv刚从Conv2d切换而来）
            params_to_optimize.append({
                'params': arconv_params,
                'lr': args.learning_rate * 0.1,
                'name': 'arconv'
            })
            print(f"✅ ARConv参数: {len(arconv_params)} 个（lr=0.1x）")
        elif stage in ["full", "fixed"]:
            # Full/Fixed阶段：正常学习率
            # 注意：fixed阶段offset/modulation虽然固定，但weight仍在训练
            params_to_optimize.append({
                'params': arconv_params,
                'lr': args.learning_rate * 0.5,
                'name': 'arconv'
            })
            print(f"✅ ARConv参数: {len(arconv_params)} 个（lr=0.5x）")
    
    # model_reg的参数
    reg_params = [p for p in model_reg.parameters() if p.requires_grad]
    if len(reg_params) > 0:
        params_to_optimize.append({
            'params': reg_params,
            'lr': args.learning_rate,
            'name': 'reg'
        })
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )
    
    return optimizer


def print_stage_transition(global_step, stage, next_stage, args):
    """
    打印阶段转换信息
    
    Args:
        global_step: 当前步数
        stage: 当前阶段
        next_stage: 下一阶段
        args: 训练参数
    """
    if stage == next_stage:
        return
    
    print("\n" + "="*70)
    print(f"🔄 [步数 {global_step}] 训练阶段切换: {stage} → {next_stage}")
    print("="*70)
    
    if next_stage == "warmup":
        print("📌 阶段1: Warmup (使用Conv2d)")
        print("   📦 架构: Conv2d（继承预训练+LoRA）")
        print("   🔥 LoRA: 训练中")
        print("   🔥 SceneToken: 训练中")
        print("   📊 目标: 让Student学习Teacher的基础能力")
    
    elif next_stage == "activate":
        warmup_steps = getattr(args, 'arconv_warmup_steps', 100)
        print(f"📌 阶段2: ARConv Activate (从第{warmup_steps}步开始)")
        print("   🔄 架构: Conv2d → ARConv（插值扩展初始化）")
        print("   🔥 ARConv: 训练所有参数（lr=0.1x）")
        print("       - offset/modulation: 从零开始学习自适应配置")
        print("       - weight: 从Conv2d权重初始化并继续训练")
        print("   🔥 LoRA: 训练中")
        print("   🔥 SceneToken: 训练中")
        print("   📊 目标: ARConv逐步发挥自适应能力")
    
    elif next_stage == "full":
        print("📌 阶段3: Full Training")
        print("   🔥🔥 ARConv: 全力训练（lr=0.5x）")
        print("       - offset/modulation: 持续优化自适应配置")
        print("       - weight: 全力训练")
        print("   🔥 LoRA: 训练中")
        print("   🔥 SceneToken: 训练中")
        print("   📊 目标: 三模块协同优化，达到最优性能")
    
    elif next_stage == "fixed":
        fixstep = getattr(args, 'arconv_fixstep', 4000)
        print(f"📌 阶段4: Fixed Configuration (fixstep={fixstep})")
        print("   🔥 ARConv: 继续训练（lr=0.5x）")
        print("       - offset/modulation: 🔒 固定（不再变化）")
        print("       - weight: 🔥 继续训练")
        print("   🔥 LoRA: 训练中")
        print("   🔥 SceneToken: 训练中")
        print("   📊 目标: 在固定的卷积核配置上继续优化权重")
        print("   ℹ️  注意: ARConv的自适应配置已固定为最优形状，")
        print("           但卷积权重仍在学习，模型继续提升！")
    
    print("="*70 + "\n")


def should_update_optimizer(global_step, last_update_step, args):
    """
    判断是否需要更新优化器（阶段切换时）
    
    Args:
        global_step: 当前步数
        last_update_step: 上次更新优化器的步数
        args: 训练参数
    
    Returns:
        bool: 是否需要更新
    """
    warmup_steps = getattr(args, 'arconv_warmup_steps', 100)
    activate_steps = getattr(args, 'arconv_activate_steps', 3000)
    scene_token_freeze_steps = getattr(args, 'scene_token_freeze_steps', 300)
    # fixstep不需要更新优化器，因为ARConv参数仍然可训练
    
    # 在关键节点需要更新优化器
    critical_steps = [warmup_steps, activate_steps, scene_token_freeze_steps]
    
    return global_step in critical_steps and global_step != last_update_step


def get_stage_info(global_step, args):
    """
    获取当前阶段的详细信息
    
    Returns:
        dict: 阶段信息
    """
    stage = get_training_stage(global_step, args)
    warmup_steps = getattr(args, 'arconv_warmup_steps', 2000)
    activate_steps = getattr(args, 'arconv_activate_steps', 3000)
    fixstep = getattr(args, 'arconv_fixstep', 4000)
    
    info = {
        'stage': stage,
        'arconv_enabled': stage != "warmup",
        'arconv_lr_multiplier': {
            'warmup': 0.0,
            'activate': 0.1,
            'full': 0.5,
            'fixed': 0.5
        }[stage],
        'offset_modulation_trainable': stage != "fixed",  # fixstep后固定
        'description': {
            'warmup': f"Warmup (步数 0-{warmup_steps})",
            'activate': f"ARConv Activate (步数 {warmup_steps}-{activate_steps})",
            'full': f"Full Training (步数 {activate_steps}-{fixstep})",
            'fixed': f"Fixed Configuration (步数 {fixstep}+)"
        }[stage]
    }
    
    return info

