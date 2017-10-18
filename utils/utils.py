def tee(text, output_file):
    """
    Prints text to standard output and also writes it to a file.
    :param text: String to print and write.
    :param output_file: File to write to.
    """
    print(text)
    output_file.write(text + '\n')
