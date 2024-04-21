import db_tool
import itertools
from collections import Counter
import pyecharts.options as opts
from pyecharts.charts import WordCloud
import pandas
from translate import Translator


def generate_word_cloud(tag, file_path, keywords):
    translator = Translator(from_lang='ZH', to_lang='EN')
    session = db_tool.MongoSessionHelper()
    all_datas = session.get_all_data(db_name='bill_intangible_cultural_heritage_db', collection_name='message_word_cut')
    culture_datas = [o['word_cut'] for o in all_datas if o['tag'] == tag]
    culture_word_dicts = list(itertools.chain.from_iterable(culture_datas))
    culture_words = [o['word'] for o in culture_word_dicts if len(o['word']) > 1 and 'n' in o['pos']]
    culture_words_count = list(Counter(culture_words).items())
    culture_words_count = sorted(culture_words_count, key=lambda o: o[1], reverse=True)
    culture_words_count = [o for o in culture_words_count if o[1] > 1][:100]
    culture_words_count = [(translator.translate(o[0]), o[1]) for o in culture_words_count if o[0] not in keywords]
    print(culture_words_count)
    # save culture_words_count to csv
    culture_words_count_df = pandas.DataFrame(culture_words_count, columns=['word', 'count'])
    culture_words_count_df.to_csv(file_path, index=False)

    (
        WordCloud()
        .add(series_name="热点分析", data_pair=culture_words_count, word_size_range=[6, 66])
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="热点分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
            ),
            tooltip_opts=opts.TooltipOpts(is_show=True),
        )
        .render("wordcloud_culture.html")
    )


generate_word_cloud(tag=1, file_path='culture_words_count.csv', keywords=['非物质文化遗产', '视频', '文化遗产'])
generate_word_cloud(tag=2, file_path='social_words_count.csv', keywords=['非物质文化遗产', '视频', '文化遗产'])
generate_word_cloud(tag=3, file_path='eco_words_count.csv', keywords=['非物质文化遗产', '视频', '文化遗产'])
