CREATE SCHEMA IF NOT EXISTS psi;

CREATE TABLE IF NOT EXISTS psi.redditors (
    user_name VARCHAR(255) PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS psi.subreddits (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS psi.posts (
    id VARCHAR(255) PRIMARY KEY,
    subreddit_id VARCHAR(255) NOT NULL REFERENCES psi.subreddits(id),
    redditor VARCHAR(255) NOT NULL REFERENCES psi.redditors(user_name),
    num_upvotes INT NOT NULL,
    upvote_ratio FLOAT NOT NULL,
    time_created TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS psi.comments (
    id VARCHAR(255) PRIMARY KEY,
    post_id VARCHAR(255) NOT NULL REFERENCES psi.posts(id),
    redditor VARCHAR(255) NOT NULL REFERENCES psi.redditors(user_name),
    num_upvotes INT NOT NULL,
    time_created TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS psi.companies (
    ticker VARCHAR(255) UNIQUE,
    name VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS psi.post_sentiments (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(255) NOT NULL REFERENCES psi.posts(id),
    company_ticker VARCHAR(255),
    sentiment_score FLOAT CHECK (sentiment_score BETWEEN -1 AND 1 OR sentiment_score IS NULL),
    UNIQUE(post_id, company_ticker)
);

CREATE TABLE IF NOT EXISTS psi.comment_sentiments (
    id SERIAL PRIMARY KEY,
    comment_id VARCHAR(255) NOT NULL REFERENCES psi.comments(id),
    company_ticker VARCHAR(255) REFERENCES psi.companies(ticker),
    sentiment_score FLOAT NOT NULL CHECK (sentiment_score BETWEEN -1 AND 1 OR sentiment_score IS NULL),
    UNIQUE(comment_id, company_ticker)
);