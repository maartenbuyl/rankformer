import torch


class BaseRankLoss(torch.nn.Module):
    def forward(self, score, target):
        raise NotImplementedError

    def forward_per_list(self, score, target, length):
        # Split score and target into lists
        length_per_list = length.tolist()
        score_per_list = score.split(length_per_list)
        target_per_list = target.split(length_per_list)

        # Compute loss per list, giving each list equal weight (regardless of length)
        loss_per_list = [
            self(score_of_list, target_of_list)
            for score_of_list, target_of_list in zip(score_per_list, target_per_list)
        ]
        losses = torch.stack(loss_per_list)

        # Remove losses that are zero (e.g. all item labels are zero)
        losses = losses[torch.abs(losses) > 0.]
        if len(losses) == 0:
            # If all losses were removed, take the sum (which will result in a zero gradient)
            return losses.sum()

        loss = losses.mean()
        return loss


class MSELoss(BaseRankLoss):
    def forward(self, score, target):
        return torch.nn.functional.mse_loss(score, target)


class OrdinalLoss(BaseRankLoss):
    # See A Neural Network Approach to Ordinal Regression

    def forward(self, score, target):
        # Prepare a target column for each ordinal value
        encoded_target = torch.zeros(score.shape).to(score.device)
        for col in range(score.shape[1]):
            encoded_target[:, col] = (target > col).int()

        loss = torch.nn.functional.binary_cross_entropy_with_logits(score, encoded_target)
        return loss


class SoftmaxLoss(BaseRankLoss):
    def forward(self, score, target):
        softmax_score = torch.nn.functional.log_softmax(score, dim=-1)
        loss = -(softmax_score * target).mean()
        return loss
