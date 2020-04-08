def print_progress_bar(iteration, total, prefix="", suffix=""):
    """Call in a loop to create terminal progress bar.
    Args:
        iteration: current iteration (Int)
        total: total iterations (Int)
        prefix: prefix string (Str)
        suffix: suffix string (Str)
    """
    # Printing slowes down the loop
    if iteration % (total // 100) == 0:
        length = 40
        iteration += 1
        percent = (100 * iteration) // (total * 99 / 100)
        filled_length = int(length * percent / 100)
        bar = "█" * filled_length + "-" * (length - filled_length)
        print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")

        if iteration >= total * 99 / 100:
            print()
