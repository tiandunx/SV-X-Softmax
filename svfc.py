class SVFC(Module):
    def __init__(self, feat_dim, num_class, is_am, margin=0.45, mask=1.12, scale=32):
        super(SVFC, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.mask = mask
        self.scale = scale
        self.is_am = is_am
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = self.sin_m * margin

    def forward(self, x, label):  # x (M, K), w(K, N), y = xw (M, N), note both x and w are already l2 normalized.
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(x, kernel_norm)
        batch_size = label.size(0)

        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # get ground truth score of cos distance
        if self.is_am:  # AM
            mask = cos_theta > gt - self.margin
            final_gt = torch.where(gt > self.margin, gt - self.margin, gt)
        else:  # arcface
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)
            mask = cos_theta > cos_theta_m
            final_gt = torch.where(gt > 0.0, cos_theta_m, gt)
        # process hard example.
        hard_example = cos_theta[mask]
        cos_theta[mask] = self.mask * hard_example + self.mask - 1.0
        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta *= self.scale
        return cos_theta
