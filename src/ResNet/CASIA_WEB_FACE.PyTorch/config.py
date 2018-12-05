configurations = {
    # ResNet from scratch with SGD
    1: dict(
        # num_iter_per_epoch = ceil(num_images/batch_size)
        max_iteration=463530,  # 30 epochs on full dataset
        lr=0.1,
        weight_decay=0.0005,
        interval_validate=200,
        optim='Adam',
        batch_size=32,        
    ),

    2: dict(
        # num_iter_per_epoch = ceil(num_images/batch_size)
        max_iteration=463530,  # 30 epochs on full dataset
        lr=0.01,
        weight_decay=0.0005,
        interval_validate=200,
        optim='Adam',
        batch_size=32,        
    ),

    3: dict(
        # num_iter_per_epoch = ceil(num_images/batch_size)
        max_iteration=463530,  # 30 epochs on full dataset
        lr=0.001,
        weight_decay=0.0005,
        interval_validate=200,
        optim='Adam',
        batch_size=32,        
    ),
}
