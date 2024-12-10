import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def draw_split_processing_time_optimization():
    # 读取两个文件
    file1 = '../output/experiment/csv/frequency_influence_partial_matches_time.csv'
    file2 = '../output/experiment/csv/frequency_influence_partial_matches_independence_time.csv'

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 提取 fa 和 fb 的组合，作为颜色分配的基础
    fa_fb_combinations = pd.concat([df1[['fa', 'fb']], df2[['fa', 'fb']]]).drop_duplicates()

    # 分配颜色，每个 fa_fb 对应一个颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(fa_fb_combinations)))
    fa_fb_to_color = {tuple(row): color for row, color in zip(fa_fb_combinations.values, colors)}

    # 绘制图形
    plt.figure(figsize=(10, 6))

    # 绘制第一个文件的数据，使用虚线
    for fa_fb, color in fa_fb_to_color.items():
        fa, fb = fa_fb
        data = df1[(df1['fa'] == fa) & (df1['fb'] == fb)]
        #plt.plot(data['item_num'], data['partial_match_nums'], linestyle='--', color=color, label=f'fa={fa}, fb={fb} (dashed)')
        plt.plot(data['item_num'], data['handle_time'], linestyle='--', color=color,
                 label=f'fa={fa}, fb={fb} (dashed)')

    # 绘制第二个文件的数据，使用实线
    for fa_fb, color in fa_fb_to_color.items():
        fa, fb = fa_fb
        data = df2[(df2['fa'] == fa) & (df2['fb'] == fb)]
        plt.plot(data['item_num'], data['handle_time'], linestyle='-', color=color, label=f'fa={fa}, fb={fb} (solid)')

    # 添加图例和标签
    plt.xlabel('item_num')
    plt.ylabel('handle_time')
    plt.title('Frequency Influence on handle time (dashed vs solid)')
    plt.legend()

    # 显示图形
    plt.show()

def draw_latency_handle_processing_time_optimization():
    # 读取两个文件
    file1 = '../output/experiment/csv/frequency_influence_partial_matches_time.csv'
    file2 = '../output/experiment/csv/latency_handle_influence_matches_time.csv'

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 提取 fa 和 fb 的组合，作为颜色分配的基础
    fa_fb_combinations = pd.concat([df1[['fa', 'fb']], df2[['fa', 'fb']]]).drop_duplicates()

    # 分配颜色，每个 fa_fb 对应一个颜色
    colors = plt.cm.viridis(np.linspace(0, 1, len(fa_fb_combinations)))
    fa_fb_to_color = {tuple(row): color for row, color in zip(fa_fb_combinations.values, colors)}

    # 绘制图形
    plt.figure(figsize=(10, 6))

    # 绘制第一个文件的数据，使用虚线
    for fa_fb, color in fa_fb_to_color.items():
        fa, fb = fa_fb
        data = df1[(df1['fa'] == fa) & (df1['fb'] == fb)]
        #plt.plot(data['item_num'], data['partial_match_nums'], linestyle='--', color=color, label=f'fa={fa}, fb={fb} (dashed)')
        plt.plot(data['item_num'], data['handle_time'], linestyle='--', color=color,
                 label=f'fa={fa}, fb={fb} (dashed)')

    # 绘制第二个文件的数据，使用实线
    for fa_fb, color in fa_fb_to_color.items():
        fa, fb = fa_fb
        data = df2[(df2['fa'] == fa) & (df2['fb'] == fb)]
        plt.plot(data['item_num'], data['handle_time'], linestyle='-', color=color, label=f'fa={fa}, fb={fb} (solid)')

    # 添加图例和标签
    plt.xlabel('item_num')
    plt.ylabel('handle_time')
    plt.title('latency handle on handle time (dashed vs solid)')
    plt.legend()

    # 显示图形
    plt.show()

def plot_results_handle_time_partial_nums(without_handle_csv, with_handle_csv):
    without_handle_df = pd.read_csv(without_handle_csv).dropna(subset=['handle_time'])
    with_handle_df = pd.read_csv(with_handle_csv).dropna(subset=['handle_time'])

    # 添加NaN值来防止自动连接空的数据点
    without_handle_df['partial_match_nums'] = without_handle_df['partial_match_nums'].replace(float('inf'), np.nan)
    with_handle_df['partial_match_nums'] = with_handle_df['partial_match_nums'].replace(float('inf'), np.nan)

    frequencies = sorted(set(without_handle_df['fa']).union(set(with_handle_df['fa'])))

    colors = plt.cm.viridis(np.linspace(0, 1, len(frequencies)))

    # Plot for partial_match_nums
    plt.figure(figsize=(10, 6))
    for color, freq in zip(colors, frequencies):
        without_handle_data = without_handle_df[without_handle_df['fa'] == freq]
        with_handle_data = with_handle_df[with_handle_df['fa'] == freq]

        plt.plot(np.log(without_handle_data['item_num']), np.log(without_handle_data['partial_match_nums']), linestyle='--', color=color, label=f'without_handle fa={freq}')
        plt.plot(np.log(with_handle_data['item_num']), np.log(with_handle_data['partial_match_nums']), linestyle='-', color=color, label=f'with_handle fa={freq}')

    plt.xlabel('log(item_num)')
    plt.ylabel('log(partial_match_nums)')
    plt.title('Partial Match Numbers with and without Handle')
    plt.legend()
    plt.show()

    # Plot for handle_time
    plt.figure(figsize=(10, 6))
    for color, freq in zip(colors, frequencies):
        without_handle_data = without_handle_df[without_handle_df['fa'] == freq]
        with_handle_data = with_handle_df[with_handle_df['fa'] == freq]

        plt.plot(np.log(without_handle_data['item_num']), without_handle_data['handle_time'], linestyle='--', color=color, label=f'without_handle fa={freq}')
        plt.plot(np.log(with_handle_data['item_num']), with_handle_data['handle_time'], linestyle='-', color=color, label=f'with_handle fa={freq}')

    plt.xlabel('log(item_num)')
    plt.ylabel('log(handle_time)')
    plt.title('Handle time with and with two table join Handle')
    plt.legend()
    plt.show()
