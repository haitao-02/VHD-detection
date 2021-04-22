class PolyScheduler(object):
    def __init__(self, base_lrs, num_epochs, iters_per_epoch=0, warmup_epochs=0):
        # print('Using Poly Scheduler!')
        self.lrs = base_lrs
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch):
        T = epoch * self.iters_per_epoch + i
        for i in range(len(self.lrs)):
            lr = self.lrs[i] * pow((1 - 1.0 * T / self.N), 0.9)
            # warm up lr schedule
            if self.warmup_iters > 0 and T < self.warmup_iters:
                lr = lr * 1.0 * T / self.warmup_iters
            if epoch > self.epoch:
                self.epoch = epoch
            assert lr >= 0
            self._adjust_learning_rate(optimizer, lr, i)

    def _adjust_learning_rate(self, optimizer, lr, i):
        optimizer.param_groups[i]['lr'] = lr
