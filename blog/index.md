---
layout: default
title: Blog
nav_order: 6
has_children: false
permalink: /blog/
---

# Blog
{: .no_toc }

Latest insights on Large Language Models, AI development, and production ML.
{: .fs-6 .fw-300 }

---

<div class="posts-list">
{% for post in site.posts %}
  <div class="post-preview">
    <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
    <p class="post-meta">
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %d, %Y" }}</time>
      {% if post.author %} • by {{ post.author }}{% endif %}
      {% if post.tags.size > 0 %}
        • Tagged: 
        {% for tag in post.tags %}
          <span class="tag">{{ tag }}</span>{% unless forloop.last %}, {% endunless %}
        {% endfor %}
      {% endif %}
    </p>
    {% if post.excerpt %}
      <div class="post-excerpt">
        {{ post.excerpt | strip_html | truncatewords: 50 }}
      </div>
    {% endif %}
    <a href="{{ post.url | relative_url }}" class="read-more">Read more →</a>
  </div>
  <hr>
{% endfor %}
</div>

{% if site.posts.size == 0 %}
<div class="no-posts">
  <p>No blog posts yet. Check back soon for insights on LLMs and AI development!</p>
</div>
{% endif %}

<style>
.posts-list {
  margin-top: 2rem;
}

.post-preview {
  margin-bottom: 2rem;
}

.post-preview h2 {
  margin-bottom: 0.5rem;
}

.post-preview h2 a {
  text-decoration: none;
  color: var(--link-color);
}

.post-preview h2 a:hover {
  text-decoration: underline;
}

.post-meta {
  color: var(--body-text-color);
  opacity: 0.7;
  font-size: 0.9rem;
  margin-bottom: 1rem;
}

.tag {
  background-color: var(--code-background-color);
  padding: 0.2rem 0.4rem;
  border-radius: 3px;
  font-size: 0.8rem;
}

.post-excerpt {
  margin-bottom: 1rem;
  line-height: 1.6;
}

.read-more {
  color: var(--link-color);
  text-decoration: none;
  font-weight: 500;
}

.read-more:hover {
  text-decoration: underline;
}

.no-posts {
  text-align: center;
  padding: 3rem 0;
  color: var(--body-text-color);
  opacity: 0.7;
}
</style>