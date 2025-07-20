from typing import Annotated

from pydantic import BaseModel

from agentdojo.functions_runtime import Depends

from typing import Union

class Web(BaseModel):
    web_content: dict[str, str]
    web_requests: list[str]


AnnotatedWeb = Annotated[Web, Depends("web")]


def standardize_url(url):
    if url.startswith("https://"):
        url = url[8:]
    if url.startswith("http://"):
        url = url[7:]
    return url


def post_webpage(web: AnnotatedWeb, url: str, content: str) -> None:
    """Posts a webpage at a given URL with the given content.

    :param url: The URL of the webpage.
    :param content: The content of the webpage.
    """
    url = standardize_url(url)
    web.web_requests.append(url)
    web.web_content[url] = content

# def get_webpage(web: AnnotatedWeb, url: str) -> str | None:
#     """Returns the content of the webpage at a given URL.

#     :param url: The URL of the webpage.
#     """
#     url = standardize_url(url)
#     web.web_requests.append(url)
#     if url not in web.web_content:
#         return "404 Not Found"
#     return web.web_content.get(url)

def get_webpage(web: AnnotatedWeb, url: Union[str, list[str]]) -> dict[str|(str | None)]:
    """Returns the content of the webpage(s) at given URL(s).

    :param url: The URL(s) list or string of the webpage(s).
    """
    if isinstance(url, str):
        url = [url]

    webpages = {}
    for u in url:
        u = standardize_url(u)
        web.web_requests.append(u)
        if u not in web.web_content:
            webpages[u] = "404 Not Found"
        else:
            webpages[u] = web.web_content.get(u)
    return webpages

def download_file(web: AnnotatedWeb, url: str, file_name: str) -> str | None:
    """Downloads a file from a given URL to the local folder.

    :param url: The URL of the file.
    :param file_name: The name of the file to download.
    """
    url = standardize_url(url)
    web.web_requests.append(url)
    if url not in web.web_content:
        return "404 Not Found"
