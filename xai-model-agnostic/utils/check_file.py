def check_task_1(most_important_feature, house_age, final_prediction):
    if most_important_feature is None:
        print('Please insert only strings.')
        return
    else:
        assert type(most_important_feature)==str, 'Please insert only strings.'
    if most_important_feature.lower()=='medinc':
        print('Yes, you are right! The most important feature is the median income as it changes the prediction by about -0.4, whereas the other features have a smaller influence.')
    else:
        print('That is unfortunately not correct. If you did not mistype your answer, please have another look at the plot. You can see how much a feature influences the prediction by how much the red line moves to the left or right in the feature`s row.')
    if house_age is None:
        print('Please insert only strings.')
        return
    else:
        assert type(house_age)==str, 'Please insert only strings.'
    if house_age=='decrease':
        print('Yes, that is correct. The line moves to the left in the `HouseAge` row and therefore decreases the prediction.')
    elif house_age=='increase':
        print('No, that`s not right. Have a look at what the line is doing in the row with the label `HouseAge`. Is is going to the right (increasing the prediction) or to the left (decreating the prediction)?')
    else:
        print('Is there a typo?')
    if final_prediction is None:
        print('Please insert only strings.')
        return
    else: assert type(final_prediction)==str, 'Please insert only strings.'
    if final_prediction=='lower':
        print('Yes, true! The line ends at a value around 1.5 and that is lower than the average prediction of 1.89')
    elif final_prediction=='higher':
        print('No, not quite right. The end of the line at the top of the graph shows the final prediction.')
    else:
        print('Oh, there is a typo somewhere!')
