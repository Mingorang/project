"""Rolling-origin index generator; horizon-aware gap."""
def origins(n,min_train=100,horizon=5,step=5):
    for cutoff in range(min_train-1,n-horizon,step):
        yield {"train_end_exclusive":cutoff+1,"origin":cutoff,"test_start":cutoff+1,"test_end_exclusive":cutoff+1+horizon}
