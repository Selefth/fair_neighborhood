
import requests
import json

GENDER_PROPERTY = "P21"  # Property for gender in Wikidata
OCCUPATION_PROPERTY = "P106"  # Property for occupation in Wikidata
AUTHOR_OCCUPATION_ID = "Q36180"  # ID for the author occupation in Wikidata
MALE_GENDER_ID = "Q6581097"  # ID for male gender in Wikidata
FEMALE_GENDER_ID = "Q6581072"  # ID for female gender in Wikidata

def is_author(data):
    """Check if the person is an author."""
    if OCCUPATION_PROPERTY in data:
        for occupation in data[OCCUPATION_PROPERTY]:
            if "datavalue" in occupation["mainsnak"]:
                if occupation["mainsnak"]["datavalue"]["value"]["id"] == AUTHOR_OCCUPATION_ID:
                    return True
    return False

def get_gender(data):
    """Retrieve the gender of the person."""
    if GENDER_PROPERTY in data:
        if "datavalue" in data[GENDER_PROPERTY][0]["mainsnak"]:
            gender_id = data[GENDER_PROPERTY][0]["mainsnak"]["datavalue"]["value"]["id"]
            if gender_id == MALE_GENDER_ID:
                return "male"
            elif gender_id == FEMALE_GENDER_ID:
                return "female"
    return "unknown"

def get_author_gender(author_name):
    """Retrieve the data for the given author name from Wikidata."""
    author_name = author_name.replace(" ", "+")
    url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles={author_name}&props=claims&format=json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.RequestException:
        return None
    
    data = response.json()

    if "entities" in data:
        for entity in data["entities"].values():
            # Check if 'claims' exists in the entity and if the person is an author
            if "claims" in entity and is_author(entity["claims"]):
                return get_gender(entity["claims"])
    
    return "unknown"

def collect_authors_metadata():
    books_by_author = {}

    with open("goodreads_books_young_adult.json", "r") as books_file:
        for line in books_file:
            data = json.loads(line)
            if "authors" in data and "book_id" in data and len(data["authors"]) == 1:
                author_id = data["authors"][0]["author_id"]
                book_id = data["book_id"]
                if author_id in books_by_author:
                    books_by_author[author_id]["book_ids"].append(book_id)
                else:
                    books_by_author[author_id] = {"book_ids": [book_id]}

    with open("goodreads_books_authors.json", "r") as authors_file, \
         open("goodreads_books_young_adult_authors.json", "w") as new_file:
        for line in authors_file:
            data = json.loads(line)
            author_id = data["author_id"]
            if author_id in books_by_author:
                author_name = data.get("name", "").strip()
                if author_name:
                    author_gender = get_author_gender(author_name)
                    if author_gender in {"male", "female"}:
                        books_by_author[author_id]["name"] = author_name
                        books_by_author[author_id]["gender"] = author_gender
                        new_file.write(json.dumps({"author_id": author_id, "author_name": author_name, "author_gender": author_gender, "book_ids": books_by_author[author_id]["book_ids"]}) + "\n")

def filter_interactions_by_book_ids():
    valid_book_ids = set()
    
    # 1. Collect valid book_ids from goodreads_books_young_adult_authors.json
    with open("goodreads_books_young_adult_authors.json", "r") as authors_file:
        for line in authors_file:
            data = json.loads(line)
            if "book_ids" in data:
                valid_book_ids.update(data["book_ids"])
                
    # 2. Filter interactions and write to a new file
    with open("goodreads_interactions_young_adult.json", "r") as interactions_file, \
         open("goodreads_filtered_interactions_young_adult.json", "w") as new_file:
        for line in interactions_file:
            interaction_data = json.loads(line)
            if (interaction_data["book_id"] in valid_book_ids and 
                interaction_data["is_read"] == True and 
                interaction_data["rating"] != 0):
                new_file.write(json.dumps({"user_id": interaction_data["user_id"], "book_id": interaction_data["book_id"], "rating": interaction_data["rating"]}) + "\n")

if __name__ == "__main__":
    # collect_authors_metadata()
    filter_interactions_by_book_ids()