import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probablity_for_random_selection_on_all_pages = (1 - damping_factor) / len(corpus)
    result = {key: probablity_for_random_selection_on_all_pages for key in corpus.keys()}
    linked_pages = corpus[page]

    if len(linked_pages) == 0:
        result = {key: 1 / len(list(corpus.keys())) for key in corpus.keys()}
        return result
    
    probablity_for_random_selection_on_linked_pages = damping_factor / len(linked_pages)

    for page in linked_pages:
        result[page] += probablity_for_random_selection_on_linked_pages

    return result
    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page = random.choice(list(corpus.keys()))
    result = {key: 0 for key in corpus.keys()}
    result[page] += 1

    for i in range(n - 1):
        probablity = transition_model(corpus, page, damping_factor)
        page = weighted_choice(probablity)
        result[page] += 1
    
    for page in result:
        result[page] = result[page] / n
    
    return result
    raise NotImplementedError

def weighted_choice(choices):
    total = sum(choices.values())
    rand_val = random.uniform(0, total)
    cumulative_prob = 0

    for key, weight in choices.items():
        cumulative_prob += weight
        if rand_val < cumulative_prob:
            return key

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_pages = len(list(corpus.keys()))
    prev_probability = {key: 0 for key in corpus.keys()}
    next_probability = {key: 1 / total_pages for key in corpus.keys()}
    reverse_corpus = {key: set() for key in corpus.keys()}

    for page in corpus:
        if len(corpus[page]) == 0:
            corpus[page] = set(corpus.keys())
    
    for page in corpus:
        for linked_page in corpus[page]:
            reverse_corpus[linked_page].add(page)

    while not all(abs(prev_probability[page] - next_probability[page]) <= 0.001 for page in prev_probability):
        prev_probability.update(next_probability)
        for page in corpus:
            next_probability[page] = ((1 - damping_factor) / total_pages)
            for i_page in reverse_corpus[page]:
                next_probability[page] += damping_factor * prev_probability[i_page] / len(corpus[i_page])

    return next_probability
    raise NotImplementedError


if __name__ == "__main__":
    main()
