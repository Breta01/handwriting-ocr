def print_progress_bar(iteration,
                       total,
                       prefix = '',
                       suffix = ''):
    """Call in a loop to create terminal progress bar.
    Args:
        iteration: current iteration (Int)
        total: total iterations (Int)
        prefix: prefix string (Str)
        suffix: suffix string (Str)
        decimals: positive number of decimals in percent complete (Int)
        length: character length of bar (Int)
        fill: bar fill character (Str)
    """
    decimals = 1
    length = 40
    iteration += 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')

    if iteration == total:
        print()
