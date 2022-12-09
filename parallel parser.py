import multiprocessing.pool as mpp
from bs4 import BeautifulSoup
import multiprocessing as mp
import tqdm
import requests
import pandas as pd


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap


def get_recipe(*recipe_title):
    recipe_title = ''.join([s for s in recipe_title])
    base_url = 'https://www.epicurious.com'
    if len(recipe_title.replace(" ", "")) == 0:
        return None
    r = requests.get(f'https://www.epicurious.com/search/{recipe_title.replace(" ", "%20")}')
    soup = BeautifulSoup(r.text, "html.parser")

    recipe_title_set = set(''.join([s for s in recipe_title if s.isalpha() or s == ' ']).split())

    for card in soup.select('#react-app > span > section > div > article.recipe-content-card > header > h4 > a'):
        card_text = set(''.join([s for s in card.text if s.isalpha() or s == ' ']).split())
        if recipe_title_set.issubset(card_text):
            return base_url + card['href']
    return 'https://www.google.com/search?q=' + recipe_title.replace(' ', '+')


if __name__ == '__main__':
    data = pd.read_csv('epi_r.csv')
    data = data.dropna()

    rec_names = [s.strip() for s in data.title.to_list()]

    pool = mp.Pool(40)

    links = list(tqdm.tqdm(pool.istarmap(get_recipe, rec_names), position=0, desc="i", leave=True, colour='green', ncols=80))

    pool.close()

    data['links'] = links
    data.to_csv('epi_comp.csv')
