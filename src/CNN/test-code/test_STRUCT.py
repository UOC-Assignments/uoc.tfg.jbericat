STATS = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": [],
    "test_acc": []
}

STATS["test_acc"] = [11]

for i in range(len(STATS["test_acc"])):
    print(' \nFor epoch', i+1,'the TEST accuracy over the whole TEST dataset is %d %%' % (STATS["test_acc"][i])) 
