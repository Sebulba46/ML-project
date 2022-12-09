from recipes import SimilarRecipes, NutritionFacts, Forecast, check_ings, show_daily_menu
from translate import Translator
import sys

if __name__ == '__main__':
    ingredients = []

    try:
        _ = sys.argv[1]
        for arg in sys.argv[1:]:
            ingredients.append(arg.replace(',', ''))

    except IndexError:
        print('Введите ингредиенты по одному, после последнего введите "стоп" без ковычек:')
        while True:
            ing = input()
            translator = Translator(to_lang="en", from_lang='ru')
            translation = translator.translate(ing)

            if translation in check_ings:
                ingredients.append(translation)
            elif ing == 'стоп':
                break
            else:
                print('Этого ингредиента нет в системе')

    if len(ingredients) == 0:
        print('Вы не ввели ингредиентов. Перезапустите программу и попробуйте снова.')
        quit()

    sim_rec = SimilarRecipes(ingredients)
    find_output = sim_rec.find_all()

    if str(type(find_output)) == "<class 'pandas.core.indexes.numeric.Int64Index'>":

        num = input('Введите сколько похожих рецептов вам показать: ')
        try:
            num = int(num)
        except ValueError:
            print('Вы ввели не целое число')
            quit()

        print('\nI. НАШ ПРОГНОЗ\n')
        pred = Forecast(ingredients)
        pred.preprocess()
        print(pred.predict_rating_category())

        nut_facts = NutritionFacts(ingredients)
        nut_facts.retrieve()
        print(nut_facts.filter(n=5))

        print(f'\nIII. ТОП-{num} ПОХОЖИХ РЕЦЕПТА:\n')
        print(sim_rec.top_similar(num))

    elif find_output == -2:
        print('\nI. НАШ ПРОГНОЗ\n ')
        pred = Forecast(ingredients)
        pred.preprocess()
        print(pred.predict_rating_category())

        nut_facts = NutritionFacts(ingredients)
        nut_facts.retrieve()
        print(nut_facts.filter(n=5))

        print('\nIII. ТОП-3 ПОХОЖИХ РЕЦЕПТА:\n')
        print('Похожих рецептов не найдено')

    show_daily_menu()

