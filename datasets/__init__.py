import importlib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def find_dataset_using_name(dataset_name,phase):
    dataset_filename = "datasets." + dataset_name + "list"
    datasetlib = importlib.import_module(dataset_filename)
    dataset = None
    target_dataset_name = f"Pis{phase}{dataset_name}List"
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, Dataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))
    return dataset

def build_train_dataloader(cfg,num_processes):
    TrainDataset = find_dataset_using_name(cfg.INPUT.DATA_MODE, "train")
    train_data = TrainDataset(
        root_dir=cfg.INPUT.ROOT_DIR,
        gt_img_size=cfg.INPUT.GT.IMG_SIZE,
        pose_img_size=cfg.INPUT.POSE.IMG_SIZE,
        cond_img_size=cfg.INPUT.COND.IMG_SIZE,
        min_scale=cfg.INPUT.COND.MIN_SCALE,
        log_aspect_ratio=cfg.INPUT.COND.PRED_ASPECT_RATIO,
        pred_ratio=cfg.INPUT.COND.PRED_RATIO,
        pred_ratio_var=cfg.INPUT.COND.PRED_RATIO_VAR,
        psz=cfg.INPUT.COND.MASK_PATCH_SIZE,
        cond_img_type=cfg.INPUT.COND.IMG_TYPE, use_clip=cfg.MODEL.USE_CLIP,
        seqlen=cfg.INPUT.SEQ_LENGTH
    )
    train_loader = DataLoader(
        train_data,
        cfg.INPUT.BATCH_SIZE // num_processes // cfg.ACCELERATE.GRADIENT_ACCUMULATION_STEPS,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.INPUT.NUM_WORKERS,
        pin_memory=False
    )
    return train_loader

def build_test_loader(cfg):
    TestDataset = find_dataset_using_name(cfg.INPUT.DATA_MODE, "test")
    RealDataset = find_dataset_using_name(cfg.INPUT.DATA_MODE, "real")
    test_data = TestDataset(
        cfg.INPUT.ROOT_DIR, cfg.INPUT.GT.IMG_SIZE, cfg.INPUT.POSE.IMG_SIZE,
        cfg.INPUT.COND.IMG_SIZE, cfg.TEST.IMG_SIZE, cond_img_type=cfg.INPUT.COND.IMG_TYPE,use_clip=cfg.MODEL.USE_CLIP,
        seqlen=cfg.INPUT.SEQ_LENGTH)
    test_loader = DataLoader(
        test_data,
        cfg.TEST.MICRO_BATCH_SIZE,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=True
    )

    fid_real_data = RealDataset(cfg.INPUT.ROOT_DIR, cfg.TEST.IMG_SIZE)
    fid_real_loader = DataLoader(
        fid_real_data,
        cfg.TEST.MICRO_BATCH_SIZE,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=True
    )
    return test_loader, fid_real_loader, test_data, fid_real_data
