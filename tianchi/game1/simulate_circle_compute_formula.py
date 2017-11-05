
import sklearn as sk

def data_generator():
    pass


def linear_model(df_circle):
    feature_names = ['sim_diameter']
    x_train,x_test,y_train,y_test = sk.train_and_split(df_circle[feature_names],
                                                    df_circle['sim_cirumference'],
                                                    train_size=0.8, random_state=1)
    print(type(x_test))
    print(x_train.shape,y_train.shape)
    lr = sk.LinearRegression()
    model = lr.fit(x_train,y_train)
    print(model)
    print(lr.coef_, lr.intercept_)

    df_circle.loc[:,'pred_circumference'] =




if __name__ == "__main__":
    main()