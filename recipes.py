import random
from translate import Translator
import requests
import json
import pandas as pd
from joblib import load


class Forecast:
    """
    Предсказание рейтинга блюда или его класса
    """

    def __init__(self, list_of_ingredients):
        """
        Добавь сюда любые поля и строчки кода, которые тебе покажутся нужными.
        """
        self.ingredients = list_of_ingredients
        self.vector = None

    def preprocess(self):
        """
        Этот метод преобразует список ингредиентов в структуры данных,
        которые используются в алгоритмах машинного обучения, чтобы сделать предсказание.
        """

        df_cols = ['almond', 'amaretto', 'anchovy', 'anise', 'aperitif', 'apple', 'apple juice', 'apricot', 'artichoke',
                   'arugula', 'asian pear', 'asparagus', 'avocado', 'bacon', 'banana', 'barley', 'basil', 'bass',
                   'bean', 'beef', 'beef rib', 'beef shank', 'beef tenderloin', 'beer', 'beet', 'bell pepper', 'berry',
                   'biscuit', 'bitters', 'blackberry', 'blue cheese', 'blueberry', 'bok choy', 'bourbon', 'braise',
                   'bran', 'brandy', 'bread', 'breadcrumbs', 'brie', 'brine', 'brisket', 'broccoli', 'broccoli rabe',
                   'broil', 'brown rice', 'brownie', 'brunch', 'brussel sprout', 'buffalo', 'bulgur', 'butter',
                   'buttermilk', 'butternut squash', 'butterscotch/caramel', 'cabbage', 'calvados', 'campari', 'candy',
                   'candy thermometer', 'cantaloupe', 'capers', 'cardamom', 'carrot', 'cashew', 'casserole/gratin',
                   'cauliflower', 'caviar', 'celery', 'chambord', 'chard', 'chartreuse', 'cheddar', 'cheese', 'cherry',
                   'chestnut', 'chicken', 'chickpea', 'chile pepper', 'chili', 'chive', 'chocolate', 'cilantro',
                   'cinnamon', 'citrus', 'clam', 'clove', 'cobbler/crumble', 'coconut', 'cod', 'coffee',
                   'coffee grinder', 'cognac/armagnac', 'collard greens', 'condiment', 'condiment/spread', 'coriander',
                   'corn', 'cottage cheese', 'couscous', 'crab', 'cranberry', 'cranberry sauce', 'cream cheese',
                   'créme de cacao', 'crêpe', 'cucumber', 'cumin', 'currant', 'curry', 'custard', 'dill', 'dinner',
                   'dried fruit', 'duck', 'egg', 'endive', 'escarole', 'fennel', 'feta', 'fig', 'fish', 'flat bread',
                   'fontina', 'fortified wine', 'frangelico', 'fruit juice', 'garlic', 'gin', 'ginger', 'goat cheese',
                   'goose', 'gouda', 'gourmet', 'grains', 'grand marnier', 'granola', 'grape', 'grapefruit', 'grappa',
                   'green bean', 'green onion/scallion', 'ground beef', 'ground lamb', 'guam', 'guava', 'haiti',
                   'halibut', 'halloween', 'ham', 'hamburger', 'hanukkah', 'harpercollins', 'hawaii', 'hazelnut',
                   'healdsburg', 'healthy', 'herb', 'high fiber', 'hollywood', 'hominy/cornmeal/masa', 'honey',
                   'honeydew', "hors d'oeuvre", 'horseradish', 'hot drink', 'hot pepper', 'house & garden', 'houston',
                   'hummus', 'ice cream', 'ice cream machine', 'iced coffee', 'iced tea', 'idaho', 'illinois',
                   'indiana', 'iowa', 'ireland', 'israel', 'italy', 'jalapeño', 'jam or jelly', 'jamaica', 'japan',
                   'jerusalem artichoke', 'juicer', 'jícama', 'kahlúa', 'kale', 'kansas', 'kansas city', 'kentucky',
                   'kentucky derby', 'kid-friendly', 'kidney friendly', 'kirsch', 'kitchen olympics', 'kiwi', 'kosher',
                   'kosher for passover', 'kumquat', 'kwanzaa', 'labor day', 'lamb', 'lamb chop', 'lamb shank',
                   'lancaster', 'las vegas', 'lasagna', 'leafy green', 'leek', 'legume', 'lemon', 'lemon juice',
                   'lemongrass', 'lentil', 'lettuce', 'lima bean', 'lime', 'lime juice', 'lingonberry', 'liqueur',
                   'lobster', 'london', 'long beach', 'los angeles', 'louisiana', 'louisville', 'low cal', 'low carb',
                   'low fat', 'low sodium', 'low sugar', 'low/no sugar', 'lunar new year', 'lunch', 'lychee',
                   'macadamia nut', 'macaroni and cheese', 'maine', 'mandoline', 'mango', 'maple syrup', 'mardi gras',
                   'margarita', 'marinade', 'marinate', 'marsala', 'marscarpone', 'marshmallow', 'martini', 'maryland',
                   'massachusetts', 'mayonnaise', 'meat', 'meatball', 'meatloaf', 'melon', 'mexico', 'mezcal', 'miami',
                   'michigan', 'microwave', 'midori', 'milk/cream', 'minneapolis', 'minnesota', 'mint', 'mississippi',
                   'missouri', 'mixer', 'molasses', 'monterey jack', 'mortar and pestle', 'mozzarella', 'muffin',
                   'mushroom', 'mussel', 'mustard', 'mustard greens', 'nancy silverton', 'nectarine', 'new hampshire',
                   'new jersey', 'new mexico', 'new orleans', "new year's day", "new year's eve", 'new york',
                   'no sugar added', 'no-cook', 'noodle', 'north carolina', 'nut', 'nutmeg', 'oat', 'oatmeal',
                   'octopus', 'ohio', 'oklahoma', 'okra', 'oktoberfest', 'olive', 'omelet', 'one-pot meal', 'onion',
                   'orange', 'orange juice', 'oregano', 'orzo', 'papaya', 'paprika', 'parade', 'paris', 'parmesan',
                   'parsley', 'parsnip', 'passion fruit', 'passover', 'pea', 'peach', 'peanut', 'peanut butter', 'pear',
                   'pecan', 'pepper', 'persimmon', 'pickles', 'pine nut', 'pineapple', 'pistachio', 'pizza', 'plantain',
                   'plum', 'poblano', 'pomegranate', 'pomegranate juice', 'poppy', 'pork', 'pork chop', 'pork rib',
                   'pork tenderloin', 'potato', 'prosciutto', 'prune', 'pumpkin', 'punch', 'purim', 'quail', 'quiche',
                   'quince', 'rabbit', 'rack of lamb', 'radicchio', 'radish', 'raisin', 'raspberry', 'red wine',
                   'rhubarb', 'rice', 'ricotta', 'root vegetable', 'rosemary', 'rosh hashanah/yom kippur', 'rosé',
                   'rub', 'rum', 'rutabaga', 'rye', 'saffron', 'sake', 'salmon', 'salsa', 'sangria', 'santa monica',
                   'sardine', 'sauce', 'sausage', 'sauté', 'scallop', 'scotch', 'seed', 'semolina', 'sesame',
                   'sesame oil', 'shallot', 'shellfish', 'sherry', 'shrimp', 'sorbet', 'soufflé/meringue', 'sour cream',
                   'sourdough', 'soy', 'soy sauce', 'sparkling wine', 'spice', 'spinach', 'spirit', 'squash', 'squid',
                   'steak', 'strawberry', 'sugar conscious', 'sugar snap pea', 'sukkot', 'sweet potato/yam',
                   'swiss cheese', 'swordfish', 'tamarind', 'tangerine', 'tapioca', 'tarragon', 'tart', 'tea',
                   'tequila', 'thyme', 'tilapia', 'tofu', 'tomatillo', 'tomato', 'tortillas', 'tree nut', 'triple sec',
                   'tropical fruit', 'trout', 'tuna', 'turnip', 'vanilla', 'veal', 'vegan', 'vegetable', 'venison',
                   'vinegar', 'vodka', 'walnut', 'wasabi', 'watercress', 'watermelon', 'whiskey', 'white wine',
                   'whole wheat', 'wild rice', 'wine', 'yellow squash', 'yogurt', 'yuca', 'zucchini', 'turkey']

        df_cols = dict.fromkeys(df_cols, [0])

        for ing in self.ingredients:
            df_cols[ing] = [1]

        vector = pd.DataFrame(df_cols)

        self.vector = vector

        return vector

    def predict_rating_category(self):
        """
        Этот метод возвращает рейтинг для списка ингредиентов, используя регрессионную модель,
        которая была обучена заранее. Помимо самого рейтинга, метод также возвращает текст,
        который дает интерпретацию этого рейтинга и дает рекомендацию, как в примере выше.
        """

        transcript = {'bad': 'Невкусное.\nХоть конкретно вам может быть и понравится блюдо из этих ингредиентов, но,'
                             ' на наш взгляд, это плохая идея – готовить блюдо из них.\nХотели предупредить.',
                      'so-so': 'Приемлемое.\nВам скорее всего понравится, но ничего особенного в блюде из этих'
                               ' ингредиентов нет',
                      'great': 'Вкусное.\nВам очень понравится блюдо из этих ингредиентов.'}

        model = load('vote_model.sav')

        rating = model.predict(self.vector)[0]
        text = transcript[rating]

        return text


def show_daily_menu():
    df_menu = pd.read_csv('epi_links.csv')
    df_reg = pd.read_csv('epi_r.csv')

    df_menu['total_nuts'] = df_menu.protein + df_menu.calories + df_menu.sodium
    trans = {'breakfast': 'ЗАВТРАК', 'lunch': 'ОБЕД', 'dinner': 'УЖИН'}

    for meal in ['breakfast', 'lunch', 'dinner']:
        print(f'\n    {trans[meal]}\n        -----------------------')

        dishes = df_menu[df_menu[meal] == 1].sort_values(by='total_nuts').nlargest(5, columns=['total_nuts'])
        ind = random.choice(dishes.index)
        print(f'        {dishes.loc[[ind]].title.values[0].strip()} (рейтинг: {dishes.loc[[ind]].rating.values[0]})')
        print(f'Ингредиенты')

        data_with_ing = dishes.loc[[ind]].drop(columns=['rating', 'Unnamed: 0.1', 'Unnamed: 0',
                                                    'title', 'rating', 'calories', 'protein',
                                                    'fat', 'sodium'], errors='ignore')

        ingredients = [c for c in data_with_ing if data_with_ing[c].isin([1]).any()]

        for ing in ingredients:
            print(f'        - {ing}')

        print('\n        Nutrients:')
        for nut in ['calories', 'protein', 'fat', 'sodium']:
            print(f'        - {nut}: {dishes.loc[[ind]][nut].values[0]}%')

        print(f'\n        URL: {dishes.loc[[ind]].links.values[0]}')


class NutritionFacts:
    """
    Выдает информацию о пищевой ценности ингредиентов.
    """
    def __init__(self, list_of_ingredients):
        """
        Добавь сюда любые поля и строчки кода, которые тебе покажутся нужными.
        """
        self.ingredients = list_of_ingredients
        self.facts = None

    def retrieve(self):
        """
        Этот метод получает всю имеющуюся информацию о пищевой ценности из файла с заранее собранной информацией по заданным ингредиентам.
        Он возвращает ее в том виде, который вам кажется наиболее удобным и подходящим.
        """
        facts = dict()
        for foodName in self.ingredients:
            facts[foodName] = json.loads(requests.get('https://api.nal.usda.gov/fdc/v1/foods/search?api_key={}&query={}'.format(
                'Teb33rn88p60JBGJJSNKlKovZS3tJwitiWYgbD0V', foodName)).text)['foods'][0]['foodNutrients']

        self.facts = facts

        return facts

    def filter(self, must_nutrients=None, n=None):
        """
        Этот метод отбирает из всей информации о пищевой ценности только те нутриенты, которые были заданы в must_nutrients (пример в PDF-файле ниже),
        а также топ-n нутриентов с наибольшим значением дневной нормы потребления для заданного ингредиента.
        Он возвращает текст, отформатированный как в примере выше.
        """

        if must_nutrients is None:
            must_nutrients = ['Protein', 'Total lipid (fat)', 'Sodium, Na', 'Energy']

        daily_vals = {'Protein': 60, 'Total lipid (fat)': 65, 'Sodium, Na': 2300, 'Energy': 2250}

        text_with_facts = '\nII. ПИЩЕВАЯ ЦЕННОСТЬ\n\n'
        translator = Translator(to_lang='ru', from_lang='en')

        for name in self.facts.keys():
            text_of_nut = f'{translator.translate(name)}\n'
            if n is None or len(sorted([s for s in self.facts[name] if 'percentDailyValue' in s.keys()], key=lambda x: x['percentDailyValue'], reverse=True)[:n]) == 0 :
                for nutrient in must_nutrients:
                    try:
                        daily_val = [s['percentDailyValue'] for s in self.facts[name] if s['nutrientName'] == nutrient][0]
                        text_of_nut += f'{nutrient} - {daily_val}% of Daily Value\n'

                    except KeyError:
                        if nutrient in ['Protein', 'Total lipid (fat)', 'Sodium, Na', 'Energy']:
                            daily_val = [s['value'] for s in self.facts[name] if s['nutrientName'] == nutrient][0]
                            text_of_nut += f'{translator.translate(nutrient)} - {round(daily_val/daily_vals[nutrient]*100, 2)}% от дневной нормы\n'
                        else:
                            print('Your nutrient doesn\'t have calculated daily value')
                            break

                    except IndexError:
                        print('There are no such nutrient')

            else:
                for nutrient in sorted([s for s in self.facts[name] if 'percentDailyValue' in s.keys()], key=lambda x: x['percentDailyValue'], reverse=True)[:n]:
                    text_of_nut += f"{translator.translate(nutrient['nutrientName'])} - {nutrient['percentDailyValue']}% от дневной нормы\n"
            text_with_facts += text_of_nut + '...\n'

        return text_with_facts[:-5]


class SimilarRecipes:
    """
    Рекомендация похожих рецептов с дополнительной информацией
    """

    def __init__(self, list_of_ingredients):
        """
        Добавь сюда любые поля и строчки кода, которые тебе покажутся нужными.
        """
        self.ingredients = list_of_ingredients
        self.data = pd.read_csv('epi_links.csv')
        self.index_sorted = None

    def find_all(self):
        """
        Этот метод возвращает список индексов рецептов, которые содержат заданный список ингредиентов.
        Если нет ни одного рецепта, содержащего все эти ингредиенты, то сделайте обработку ошибки, чтобы программа не ломалась.
        """
        if not set(self.ingredients).issubset(set(self.data.columns.to_list())):
            print('Ингредиенты некорректны. Проверьте и запустите снова.')
            return -1

        has_ingredients_index = self.data[(self.data[self.ingredients] == 1).all(axis=1)].links.index
        data_with_ing = self.data.iloc[has_ingredients_index]
        data_with_ing = data_with_ing.drop(columns=['rating', 'Unnamed: 0.1', 'Unnamed: 0',
                                                                              'title', 'rating', 'calories', 'protein',
                                                                              'fat', 'sodium'], errors='ignore')

        ingredients = [c for c in data_with_ing if data_with_ing[c].isin([1, 0]).any()]

        index_sorted = data_with_ing[ingredients].sum(axis=1)[data_with_ing[ingredients].sum(axis=1) <=
                                                              len(self.ingredients) + 10].sort_values().index

        self.index_sorted = index_sorted

        if self.data.iloc[index_sorted].empty:
            return -2

        return index_sorted

    def top_similar(self, n):
        """
        Этот метод возвращает текст, форматированный как в примере выше: с заголовком, рейтингом и URL.
        Чтобы это сделать, он вначале находит топ-n наиболее похожих рецептов с точки зрения количества дополнительных ингредиентов,
        которые потребуются в этих рецептах. Наиболее похожим будет тот, в котором не требуется никаких других ингредиентов.
        Далее идет тот, у которого появляется 1 доп. ингредиент. Далее – 2.
        Если рецепт нуждается в более, чем 5 доп. ингредиентах, то такой рецепт не выводится.
        """

        text_with_recipes = ''
        for row in self.data.iloc[self.index_sorted][['title', 'rating', 'links']].head(n).to_numpy().tolist():
            text_with_recipes += f'- {row[0]}, рейтинг: {row[1]}, URL:\n{row[2]}\n'

        return text_with_recipes


check_ings = ['almond', 'amaretto', 'anchovy', 'anise', 'aperitif', 'apple', 'apple juice', 'apricot', 'artichoke',
                   'arugula', 'asian pear', 'asparagus', 'avocado', 'bacon', 'banana', 'barley', 'basil', 'bass',
                   'bean', 'beef', 'beef rib', 'beef shank', 'beef tenderloin', 'beer', 'beet', 'bell pepper', 'berry',
                   'biscuit', 'bitters', 'blackberry', 'blue cheese', 'blueberry', 'bok choy', 'bourbon', 'braise',
                   'bran', 'brandy', 'bread', 'breadcrumbs', 'brie', 'brine', 'brisket', 'broccoli', 'broccoli rabe',
                   'broil', 'brown rice', 'brownie', 'brunch', 'brussel sprout', 'buffalo', 'bulgur', 'butter',
                   'buttermilk', 'butternut squash', 'butterscotch/caramel', 'cabbage', 'calvados', 'campari', 'candy',
                   'candy thermometer', 'cantaloupe', 'capers', 'cardamom', 'carrot', 'cashew', 'casserole/gratin',
                   'cauliflower', 'caviar', 'celery', 'chambord', 'chard', 'chartreuse', 'cheddar', 'cheese', 'cherry',
                   'chestnut', 'chicken', 'chickpea', 'chile pepper', 'chili', 'chive', 'chocolate', 'cilantro',
                   'cinnamon', 'citrus', 'clam', 'clove', 'cobbler/crumble', 'coconut', 'cod', 'coffee',
                   'coffee grinder', 'cognac/armagnac', 'collard greens', 'condiment', 'condiment/spread', 'coriander',
                   'corn', 'cottage cheese', 'couscous', 'crab', 'cranberry', 'cranberry sauce', 'cream cheese',
                   'créme de cacao', 'crêpe', 'cucumber', 'cumin', 'currant', 'curry', 'custard', 'dill', 'dinner',
                   'dried fruit', 'duck', 'egg', 'endive', 'escarole', 'fennel', 'feta', 'fig', 'fish', 'flat bread',
                   'fontina', 'fortified wine', 'frangelico', 'fruit juice', 'garlic', 'gin', 'ginger', 'goat cheese',
                   'goose', 'gouda', 'gourmet', 'grains', 'grand marnier', 'granola', 'grape', 'grapefruit', 'grappa',
                   'green bean', 'green onion/scallion', 'ground beef', 'ground lamb', 'guam', 'guava', 'haiti',
                   'halibut', 'halloween', 'ham', 'hamburger', 'hanukkah', 'harpercollins', 'hawaii', 'hazelnut',
                   'healdsburg', 'healthy', 'herb', 'high fiber', 'hollywood', 'hominy/cornmeal/masa', 'honey',
                   'honeydew', "hors d'oeuvre", 'horseradish', 'hot drink', 'hot pepper', 'house & garden', 'houston',
                   'hummus', 'ice cream', 'ice cream machine', 'iced coffee', 'iced tea', 'idaho', 'illinois',
                   'indiana', 'iowa', 'ireland', 'israel', 'italy', 'jalapeño', 'jam or jelly', 'jamaica', 'japan',
                   'jerusalem artichoke', 'juicer', 'jícama', 'kahlúa', 'kale', 'kansas', 'kansas city', 'kentucky',
                   'kentucky derby', 'kid-friendly', 'kidney friendly', 'kirsch', 'kitchen olympics', 'kiwi', 'kosher',
                   'kosher for passover', 'kumquat', 'kwanzaa', 'labor day', 'lamb', 'lamb chop', 'lamb shank',
                   'lancaster', 'las vegas', 'lasagna', 'leafy green', 'leek', 'legume', 'lemon', 'lemon juice',
                   'lemongrass', 'lentil', 'lettuce', 'lima bean', 'lime', 'lime juice', 'lingonberry', 'liqueur',
                   'lobster', 'london', 'long beach', 'los angeles', 'louisiana', 'louisville', 'low cal', 'low carb',
                   'low fat', 'low sodium', 'low sugar', 'low/no sugar', 'lunar new year', 'lunch', 'lychee',
                   'macadamia nut', 'macaroni and cheese', 'maine', 'mandoline', 'mango', 'maple syrup', 'mardi gras',
                   'margarita', 'marinade', 'marinate', 'marsala', 'marscarpone', 'marshmallow', 'martini', 'maryland',
                   'massachusetts', 'mayonnaise', 'meat', 'meatball', 'meatloaf', 'melon', 'mexico', 'mezcal', 'miami',
                   'michigan', 'microwave', 'midori', 'milk/cream', 'minneapolis', 'minnesota', 'mint', 'mississippi',
                   'missouri', 'mixer', 'molasses', 'monterey jack', 'mortar and pestle', 'mozzarella', 'muffin',
                   'mushroom', 'mussel', 'mustard', 'mustard greens', 'nancy silverton', 'nectarine', 'new hampshire',
                   'new jersey', 'new mexico', 'new orleans', "new year's day", "new year's eve", 'new york',
                   'no sugar added', 'no-cook', 'noodle', 'north carolina', 'nut', 'nutmeg', 'oat', 'oatmeal',
                   'octopus', 'ohio', 'oklahoma', 'okra', 'oktoberfest', 'olive', 'omelet', 'one-pot meal', 'onion',
                   'orange', 'orange juice', 'oregano', 'orzo', 'papaya', 'paprika', 'parade', 'paris', 'parmesan',
                   'parsley', 'parsnip', 'passion fruit', 'passover', 'pea', 'peach', 'peanut', 'peanut butter', 'pear',
                   'pecan', 'pepper', 'persimmon', 'pickles', 'pine nut', 'pineapple', 'pistachio', 'pizza', 'plantain',
                   'plum', 'poblano', 'pomegranate', 'pomegranate juice', 'poppy', 'pork', 'pork chop', 'pork rib',
                   'pork tenderloin', 'potato', 'prosciutto', 'prune', 'pumpkin', 'punch', 'purim', 'quail', 'quiche',
                   'quince', 'rabbit', 'rack of lamb', 'radicchio', 'radish', 'raisin', 'raspberry', 'red wine',
                   'rhubarb', 'rice', 'ricotta', 'root vegetable', 'rosemary', 'rosh hashanah/yom kippur', 'rosé',
                   'rub', 'rum', 'rutabaga', 'rye', 'saffron', 'sake', 'salmon', 'salsa', 'sangria', 'santa monica',
                   'sardine', 'sauce', 'sausage', 'sauté', 'scallop', 'scotch', 'seed', 'semolina', 'sesame',
                   'sesame oil', 'shallot', 'shellfish', 'sherry', 'shrimp', 'sorbet', 'soufflé/meringue', 'sour cream',
                   'sourdough', 'soy', 'soy sauce', 'sparkling wine', 'spice', 'spinach', 'spirit', 'squash', 'squid',
                   'steak', 'strawberry', 'sugar conscious', 'sugar snap pea', 'sukkot', 'sweet potato/yam',
                   'swiss cheese', 'swordfish', 'tamarind', 'tangerine', 'tapioca', 'tarragon', 'tart', 'tea',
                   'tequila', 'thyme', 'tilapia', 'tofu', 'tomatillo', 'tomato', 'tortillas', 'tree nut', 'triple sec',
                   'tropical fruit', 'trout', 'tuna', 'turnip', 'vanilla', 'veal', 'vegan', 'vegetable', 'venison',
                   'vinegar', 'vodka', 'walnut', 'wasabi', 'watercress', 'watermelon', 'whiskey', 'white wine',
                   'whole wheat', 'wild rice', 'wine', 'yellow squash', 'yogurt', 'yuca', 'zucchini', 'turkey']

if __name__ == '__main__':
    show_daily_menu()