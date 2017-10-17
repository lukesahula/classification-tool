def tee(text, output_file):
    print(text)
    output_file.write(text + '\n')
