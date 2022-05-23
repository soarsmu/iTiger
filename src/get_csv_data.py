import pandas as pd

if __name__ == '__main__':
    for file_type in ['train', 'valid', 'test']:
        body_file = '../data/body.{}.txt'.format(file_type)
        title_file = '../data/title.{}.txt'.format(file_type)

        bodies, titles = [], []
        with open(body_file) as f:
            lines = f.readlines()
        for line in lines:
            bodies.append(line.strip())

        with open(title_file) as f:
            lines = f.readlines()
        for line in lines:
            titles.append(line.strip())
        
        pd.DataFrame({'text': bodies, 'summary': titles}).to_csv('../data/{}.csv'.format(file_type))