FROM astral/uv:python3.12-bookworm-slim

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        git \
        latexmk \
        lmodern \
        poppler-utils \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-latex-recommended \
    && rm -rf /var/lib/apt/lists/*
