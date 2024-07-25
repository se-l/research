import os

from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from shared.modules.logger import logger


def render_html(render_args: list, article_nm: str):
    dir_file = Path(__file__).resolve().parents[0]
    # Render the HTML template with the dynamic content
    env = Environment(loader=FileSystemLoader(dir_file))
    template = env.get_template('article_template.html')
    rendered_html = template.render(items=render_args)

    # Save the rendered HTML to a file
    path_article = os.path.join(dir_file, f"{article_nm}.html")
    with open(path_article, 'w', encoding='utf-8') as f:
        f.write(rendered_html)
    logger.info(f'Saved article to {path_article}')

