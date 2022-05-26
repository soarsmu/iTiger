"""
Analyze the human evaluation
"""

import pandas as pd
import ast
import os

# itape, bart

def convert_real_order():
    preference_count = {'bart': 0, 'itape': 0}

    for file_name in os.listdir('../human-evaluation/'):
        if os.path.isfile(os.path.join('../human-evaluation/', file_name)) and file_name.startswith('evaluation-'):
            print(file_name)
            if file_name == 'evaluation-1.csv':
                eval_df = pd.read_csv(os.path.join('../human-evaluation/',file_name), sep=';', index_col=0)
            else:
                eval_df = pd.read_csv(os.path.join('../human-evaluation/',file_name), index_col=0)
            ## for rank
            ground_file = os.path.join('../human-evaluation/', './sampled-ground.csv')
            df = pd.read_csv(ground_file)

            for index, row in df.iterrows():
                s = row['result'][1:-1]
                s = s.replace('}, {', ',')
                cur_map = ast.literal_eval(s)
                keys = list(cur_map.keys())
                # ['bart', 'itape']

                num_to_app = dict()
                for i in range(2):
                    num_to_app[i + 1] = keys[i]
                    cur_map[keys[i]] = i + 1

                row = eval_df.iloc[index]

                rank = eval_df.iloc[index].loc['preference']
                # print(rank)
                preference_count[num_to_app[rank]] += 1

    print(preference_count)

if __name__ == "__main__":
    convert_real_order()