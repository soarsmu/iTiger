"""
we sample 30 issues for human evaluation
"""

import pandas as pd
import random
random.seed(42)

if __name__ == '__main__':
    n = 30

    test_csv = '../data/test.csv'

    df = pd.read_csv(test_csv)

    sources, titles = df['text'].tolist(), df['summary'].tolist()
    indexes = list(range(len(sources)))

    sampled = random.sample(indexes, n)
    
    with open('sampled_index.txt', 'w') as f:
        for index in sampled:
            f.write(str(index) + '\n')
    
    with open('../predicted/itape.txt') as f:
        itape_lines = f.readlines()

    with open('../predicted/bart.txt') as f:
        bart_lines = f.readlines()

    sampled_source, sampled_gold = list(), list()
    titles_1, titles_2 = list(), list()

    for index in sampled:
        sampled_source.append(sources[index])
        two_titles = [{'bart': bart_lines[index].strip()}, {'itape': itape_lines[index].strip()}]

        random.shuffle(two_titles)
        sampled_gold.append(two_titles)

        print(two_titles)
        titles_1.append(list(three_titles[0].values())[0])
        titles_2.append(list(three_titles[1].values())[0])
        

    pd.DataFrame({
        'source': sampled_source,
        'result': sampled_gold}).to_csv('sampled-ground.csv', index=False)

    pd.DataFrame({
        'source': sampled_source,
        'title_1': titles_1,
        'title_2': titles_2}).to_csv('sampled-for-evaluation.csv', index=False)