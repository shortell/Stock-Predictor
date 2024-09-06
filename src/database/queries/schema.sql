CREATE SCHEMA IF NOT EXISTS rsi;

CREATE TABLE IF NOT EXISTS rsi.redditors (
    user_name VARCHAR(255) PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS rsi.subreddits (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS rsi.posts (
    id VARCHAR(255) PRIMARY KEY,
    subreddit_id VARCHAR(255) NOT NULL REFERENCES rsi.subreddits(id),
    redditor VARCHAR(255) NOT NULL REFERENCES rsi.redditors(user_name),
    num_upvotes INT NOT NULL,
    upvote_ratio FLOAT NOT NULL,
    time_created TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS rsi.comments (
    id VARCHAR(255) PRIMARY KEY,
    post_id VARCHAR(255) NOT NULL REFERENCES rsi.posts(id),
    redditor VARCHAR(255) NOT NULL REFERENCES rsi.redditors(user_name),
    num_upvotes INT NOT NULL,
    time_created TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS rsi.companies (
    ticker VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS rsi.post_stock_sentiments (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(255) NOT NULL REFERENCES rsi.posts(id),
    company_ticker VARCHAR(255) REFERENCES rsi.companies(ticker),
    sentiment_score FLOAT CHECK (sentiment_score BETWEEN -1 AND 1 OR sentiment_score IS NULL),
    UNIQUE(post_id, company_ticker)
);

CREATE TABLE IF NOT EXISTS rsi.comment_stock_sentiments (
    id SERIAL PRIMARY KEY,
    comment_id VARCHAR(255) NOT NULL REFERENCES rsi.comments(id),
    company_ticker VARCHAR(255) REFERENCES rsi.companies(ticker),
    sentiment_score FLOAT NOT NULL CHECK (sentiment_score BETWEEN -1 AND 1 OR sentiment_score IS NULL),
    UNIQUE(comment_id, company_ticker)
);