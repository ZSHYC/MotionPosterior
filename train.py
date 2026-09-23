import argparse
import torch
torch.backends.cudnn.benchmark = True
# from torch.optim import lr_scheduler
import torch.optim.lr_scheduler as pt_lr_scheduler
from torch.utils.data import DataLoader
import importlib.util
from pathlib import Path
import sys # 确保导入 sys
import random
import numpy as np
 
#torch.autograd.set_detect_anomaly(True)

# --- 1. 导入我们所有的“工厂”的建造函数 ---
# 导入顶层包，对应的 __init__.py 文件会确保所有模块都已注册
import models_factory
import datasets_factory
import losses_factory
import optimizers_factory
import metrics_factory
import engine

# 导入我们最终的“执行器” Runner
from engine.runner import Runner

def load_config_from_path(config_path: str):
    """从 .py 文件路径中加载配置模块。"""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
        
    # 将 .py 文件作为模块加载
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    cfg_module = importlib.util.module_from_spec(spec)
    
    # 【重要】确保模块可以被再次加载（如果Python已经缓存了同名模块）
    if config_path.stem in sys.modules:
        del sys.modules[config_path.stem]
        
    spec.loader.exec_module(cfg_module)
    return cfg_module

# ===================================================================
# 步骤 1: 将你原来的 main() 函数重构为 run_experiment()
# ===================================================================
def run_experiment(config_path_str: str):
    """
    使用单个配置文件运行一次完整的训练。
    （这就是你原来 main() 函数的主体内容）
    """
    
    try:
        print(f"\n{'='*80}")
        print(f"🚀 [STARTING] 实验: {config_path_str}")
        print(f"{'='*80}\n")
        
        # --- A. 加载配置 ---
        # 不再硬编码，而是使用传入的参数
        cfg = load_config_from_path(config_path_str)
        print("✅ Configuration loaded successfully.")
        seed = getattr(cfg, 'seed', None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        # --- B. 环境设置 (由 Runner 内部处理或在这里设置) ---
        # (保持不变)
        
        # --- C. 使用工厂按图索骥，构建所有组件 ---
        print("Building components from config...")
        
        # 构建模型
        model = models_factory.build_model(cfg.model)
        print("✅ Model built successfully.")

        # 构建数据集
        train_dataset = datasets_factory.build_dataset(cfg.data['train'])
        val_dataset = datasets_factory.build_dataset(cfg.data['val'])
        print("✅ Datasets built successfully.")
        
        # 构建数据加载器 (DataLoader)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.data['samples_per_gpu'],
            num_workers=cfg.data['workers_per_gpu'],
            shuffle=True,
            pin_memory=True,
            persistent_workers=cfg.data['workers_per_gpu'] > 0
        )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=cfg.data['samples_per_gpu'],
            num_workers=cfg.data['workers_per_gpu'],
            shuffle=False,
            pin_memory=True,
            persistent_workers=cfg.data['workers_per_gpu'] > 0
        )
        print("✅ DataLoaders built successfully.")

        # 构建损失函数
        criterion = losses_factory.build_loss(cfg.loss)
        print("✅ Loss function built successfully.")

        # 构建优化器
        optimizer = optimizers_factory.build_optimizer(model, cfg.optimizer)
        print("✅ Optimizer built successfully.")
        
        # 构建评估指标
        metric = metrics_factory.build_metric(cfg.evaluation['metric'])
        print("✅ Metric built successfully.")

        # 构建钩子 (Hooks)
        hooks = engine.build_hooks(cfg.log_config['hooks'])
        if hasattr(cfg, 'custom_hooks'):
            hooks.extend(engine.build_hooks(cfg.custom_hooks))
        print("✅ All Hooks built successfully.")

        # 构建学习率调度器
        scheduler = None
        if hasattr(cfg, 'lr_config') and cfg.lr_config is not None:
            policy = cfg.lr_config.get('policy', None)
            
            if policy == 'Step':
                if 'step' not in cfg.lr_config:
                    raise ValueError("Step policy requires 'step'(list of milestones in lr_config")
                scheduler = pt_lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=cfg.lr_config['step'],
                    gamma=cfg.lr_config.get('gamma', 0.1)
                )
                print(f"✅ LR MultiStepLR scheduler built successfully.")

            elif policy == 'CosineAnnealing':
                scheduler = pt_lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=cfg.total_epochs - cfg.lr_config.get('warmup_iters', 0),
                    eta_min=cfg.lr_config.get('min_lr', 0)
                )
                print(f"✅ LR CosineAnnealingLR scheduler built successfully.")
            else:
                print(f"⚠️ LR policy '{policy}' is not supported yet or is None. Running with a fixed learning rate.")
                scheduler = None
        else:
            print("ℹ️ No LR scheduler configured. Running with a fixed learning rate.")

        # --- D. 实例化“赛车手”(Runner) ---
        runner = Runner(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            metric=metric,
            train_loader=train_loader,
            val_loader=val_loader,
            lr_scheduler=scheduler,
            hooks=hooks,
            cfg=cfg,
            # IsAMP=True
        )
        
        # --- E. 启动训练！---
        runner.run()

        print(f"\n{'-'*80}")
        print(f"✅ [SUCCESS] 实验 {config_path_str} 完成！")
        print(f"{'-'*80}\n")

    except Exception as e:
        # 添加错误处理，确保一个实验失败后，队列中的下一个实验能继续
        print(f"\n{'!'*80}")
        print(f"❌ [FAILED] 实验 {config_path_str} 遇到错误: {e}")
        import traceback
        traceback.print_exc() # 打印完整的错误堆栈
        print(f"{'!'*80}\n")
    
    # 【重要】清理显存，为下一个实验做准备
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================================================================
# 步骤 2: 新建 main() 函数作为“实验启动器”
# ===================================================================
def main():
    """
    新的主函数：扫描配置、显示菜单并按顺序运行所选实验。
    """
    config_dir = Path('./configs/')
    
    # 1. 自动扫描 configs/official 下的所有 .py 文件
    print(f"🔍 正在扫描 {config_dir}...")
    # 使用 sorted 确保每次运行的顺序都一样
    config_paths = sorted([p for p in config_dir.glob('**/*.py') if p.name != '__init__.py'])
    
    if not config_paths:
        print(f"❌ 错误: 在 {config_dir} 中未找到任何 .py 配置文件。")
        return

    # 2. 展示一个菜单
    print("\n--- 📜 可用的配置文件 ---")
    for i, path in enumerate(config_paths, 1):
        # 使用 :>2 来右对齐序号，使其更美观
        print(f"  {i:>2}: {path.name}")
    print("---------------------------\n")

    # 3. 让你输入想运行的序号
    while True:
        selection_str = input("👉 请输入要运行的配置序号 (用空格隔开, 如 '1 3 2'): ")
        selected_indices = selection_str.split()
        
        experiments_to_run = [] # 存储 (path_str, name)
        valid_input = True
        
        if not selected_indices:
            print("⚠️ 输入为空，请至少选择一个。")
            continue
            
        try:
            for s_idx in selected_indices:
                idx = int(s_idx)
                if 1 <= idx <= len(config_paths):
                    # 索引是从1开始的，列表是从0开始的
                    selected_path = config_paths[idx - 1]
                    # 存储配置的完整路径字符串和文件名
                    experiments_to_run.append((str(selected_path), selected_path.name))
                else:
                    print(f"❌ 序号 '{idx}' 超出范围 (必须在 1 到 {len(config_paths)} 之间)。")
                    valid_input = False
                    break # 停止解析这一批输入
        except ValueError:
            print(f"❌ 输入无效: '{s_idx}' 不是一个数字。")
            valid_input = False
        
        if valid_input:
            # 确认选择
            print("\n你已选择按以下顺序执行：")
            for i, (_, name) in enumerate(experiments_to_run, 1):
                print(f"  {i}. {name}")
            confirm = input("确认执行？ (y/n): ").strip().lower()
            if confirm == 'y':
                break # 输入有效且已确认，跳出 'while True' 循环
            else:
                print("🔄 已取消，请重新输入。")
        else:
            print("🔄 请重新输入。")
            
    # 4. 按顺序执行选中的实验
    print(f"\n✨ 准备按顺序执行 {len(experiments_to_run)} 个实验...")
    for i, (path_str, name) in enumerate(experiments_to_run, 1):
        print(f"\n--- 队列: {i} / {len(experiments_to_run)} ---")
        # 调用我们重构的函数
        run_experiment(path_str)
        
    print("\n🎉 所有选定的实验均已执行完毕！")


if __name__ == '__main__':
    main() # 运行新的启动器
