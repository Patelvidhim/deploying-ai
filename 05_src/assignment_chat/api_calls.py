# api_call.py

import requests
import json
import xmltodict


def get_arxiv_info(search_query='all:electron', start=0, max_results=5):
    """Fetch ArXiv papers as a list of dicts."""

    def xml_to_dict(xml_text):
        return json.loads(json.dumps(xmltodict.parse(xml_text, process_namespaces=True)))

    def parse_entries(feed):
        entries = feed.get("http://www.w3.org/2005/Atom:feed", {}).get("http://www.w3.org/2005/Atom:entry", [])
        if isinstance(entries, dict):
            entries = [entries]

        papers = []
        for e in entries:
            authors = e.get("http://www.w3.org/2005/Atom:author", [])
            if isinstance(authors, dict):
                authors = [authors]
            authors = [{"name": a.get("http://www.w3.org/2005/Atom:name"),
                        "affiliation": a.get("http://arxiv.org/schemas/atom:affiliation", {}).get("#text")}
                       for a in authors]

            links = e.get("http://www.w3.org/2005/Atom:link", [])
            if isinstance(links, dict):
                links = [links]
            pdf_link = next((l.get("@href") for l in links if l.get("@title") == "pdf"), None)

            categories = e.get("http://www.w3.org/2005/Atom:category", [])
            if isinstance(categories, dict):
                categories = [categories]
            categories = [c["@term"] for c in categories if "@term" in c]

            papers.append({
                "title": e.get("http://www.w3.org/2005/Atom:title", "").strip(),
                "abstract": e.get("http://www.w3.org/2005/Atom:summary", "").strip(),
                "doi": e.get("http://arxiv.org/schemas/atom:doi", {}).get("#text"),
                "journal_reference": e.get("http://arxiv.org/schemas/atom:journal_ref", {}).get("#text"),
                "authors": authors,
                "primary_category": e.get("http://arxiv.org/schemas/atom:primary_category", {}).get("@term"),
                "categories": categories,
                "pdf_link": pdf_link
            })
        return papers

    url = 'http://export.arxiv.org/api/query'
    response = requests.get(url, params={'search_query': search_query, 'start': start, 'max_results': max_results})
    response.raise_for_status()

    feed_dict = xml_to_dict(response.text)
    papers = parse_entries(feed_dict)

    return papers


# Example usage
if __name__ == "__main__":
    for paper in get_arxiv_info("all:quantum computing", 0, 3):
        print(paper)
