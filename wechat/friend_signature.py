
import jieba
import itchat
import wordcloud
import re


def main():
    itchat.login()
    friends = itchat.get_friends(update=True)[0:]
    sig_list = []
    print("You have total %d friends"%len(friends))
    for f in friends:
        sig = f['Signature'].replace(" ", "").replace("span", "").replace("class", "").replace("emoji", "")
        rep = re.compile("1f\d.+")
        sig = rep.sub("",sig)
        sig_list.append(sig)
    text = "".join(sig_list)
    wordlist = jieba.cut(text,cut_all=True)
    wl_space_list = " ".join(wordlist)
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import PIL.Image as Image

    my_wc = WordCloud(background_color='white',
                      max_words=200,
                      max_font_size=40,
                      random_state=30,
                      font_path="C:\Windows\winsxs\\amd64_microsoft-windows-font-truetype-arial_31bf3856ad364e35_6.1.7601.17514_none_d0a9759ec3fa9e2d\\arial.ttf").generate(wl_space_list)
    plt.imshow(my_wc)
    plt.axis("off")
    plt.show()
    #for i in friends:



if __name__ == "__main__":
    main()