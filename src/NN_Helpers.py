# Takes a list of accuracies and a window size Nand determines whether it's time to stop training.
# It does this be determining whether the average accuracy of the last N epochs is greater 
# than average of the N epochs prior.
def finished_training(accuracies, N):
    curr = len(accuracies)
    curr_mean = sum(accuracies[curr - N:curr])/ N
    prev_mean = sum(accuracies[curr-2*N : curr-N]) / N
    if curr_mean > prev_mean:
      return False
    else:
      return True