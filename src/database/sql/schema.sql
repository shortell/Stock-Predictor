CREATE SCHEMA IF NOT EXISTS rsst;

CREATE TABLE IF NOT EXISTS rsst.redditors (
    user_name VARCHAR(255) PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS rsst.subreddits (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS rsst.posts (
    id VARCHAR(255) PRIMARY KEY,
    subreddit_id VARCHAR(255) NOT NULL REFERENCES rsst.subreddits(id),
    redditor VARCHAR(255) NOT NULL REFERENCES rsst.redditors(user_name),
    num_upvotes INT NOT NULL,
    upvote_ratio FLOAT NOT NULL,
    time_created TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS rsst.comments (
    id VARCHAR(255) PRIMARY KEY,
    post_id VARCHAR(255) NOT NULL REFERENCES rsst.posts(id),
    redditor VARCHAR(255) NOT NULL REFERENCES rsst.redditors(user_name),
    num_upvotes INT NOT NULL,
    time_created TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS rsst.companies (
    ticker VARCHAR(255) UNIQUE,
    name VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS rsst.post_sentiments (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(255) NOT NULL REFERENCES rsst.posts(id),
    company_ticker VARCHAR(255),
    sentiment_label VARCHAR(255) CHECK (sentiment_label IN ('positive', 'negative', 'neutral') OR sentiment_label IS NULL),
    sentiment_score FLOAT CHECK (sentiment_score BETWEEN 0 AND 100 OR sentiment_score IS NULL),
    UNIQUE(post_id, company_ticker)
);

CREATE TABLE IF NOT EXISTS rsst.comment_sentiments (
    id SERIAL PRIMARY KEY,
    comment_id VARCHAR(255) NOT NULL REFERENCES rsst.comments(id),
    company_ticker VARCHAR(255) REFERENCES rsst.companies(ticker),
    sentiment_label VARCHAR(255) CHECK (sentiment_label IN ('positive', 'negative', 'neutral') OR sentiment_label IS NULL),
    sentiment_score FLOAT NOT NULL CHECK (sentiment_score BETWEEN 0 AND 100 OR sentiment_score IS NULL),
    UNIQUE(comment_id, company_ticker)
);

CREATE TABLE IF NOT EXISTS rsst.labeled_post_sentiments (
    id SERIAL PRIMARY KEY,
    post_sentiment_id VARCHAR(255) NOT NULL REFERENCES rsst.posts(id)
);

CREATE TABLE IF NOT EXISTS rsst.split_posts(
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(255) NOT NULL REFERENCES rsst.posts(id),
    company_ticker VARCHAR(255),
    split VARCHAR(255)
);