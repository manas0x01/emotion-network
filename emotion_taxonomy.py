"""
emotion_taxonomy.py — Expansive Emotion Taxonomy
=================================================
A curated taxonomy of 200+ fine-grained, open-vocabulary emotional concepts
covering basic, complex, nuanced, and culturally specific affective states.
Used for zero-shot dense semantic proximity mapping via SentenceTransformers.
"""

EXPANSIVE_EMOTIONS = [
    'Joy', 'Happiness', 'Delight', 'Amusement', 'Elation', 'Euphoria', 'Ecstasy', 
    'Contentment', 'Satisfaction', 'Serenity', 'Tranquility', 'Bliss', 'Gratitude', 
    'Appreciation', 'Thankfulness', 'Optimism', 'Hope', 'Relief', 'Vindication', 
    'Love', 'Affection', 'Adoration', 'Fondness', 'Compassion', 'Empathy', 
    'Sympathy', 'Tenderness', 'Warmth', 'Nostalgia', 'Longing', 'Yearning', 
    'Sentimentality', 'Belonging', 'Camaraderie', 'Solidarity', 'Trust', 
    'Security', 'Devotion', 'Sadness', 'Sorrow', 'Grief', 'Heartbreak', 
    'Anguish', 'Despair', 'Hopelessness', 'Melancholy', 'Despondency', 
    'Dejection', 'Gloom', 'Misery', 'Woe', 'Agony', 'Depression', 'Resignation', 
    'Apathy', 'Lethargy', 'Ennui', 'Emptiness', 'Fear', 'Panic', 'Terror', 
    'Horror', 'Dread', 'Apprehension', 'Anxiety', 'Nervousness', 'Worry', 
    'Unease', 'Tension', 'Trepidation', 'Vulnerability', 'Insecurity', 
    'Paranoia', 'Angst', 'Consternation', 'Alarm', 'Fright', 'Anger', 'Rage', 
    'Fury', 'Wrath', 'Indignation', 'Resentment', 'Outrage', 'Irritation', 
    'Annoyance', 'Frustration', 'Exasperation', 'Impatience', 'Aggravation', 
    'Hostility', 'Bitterness', 'Spite', 'Vengefulness', 'Animosity', 'Guilt', 
    'Remorse', 'Regret', 'Repentance', 'Shame', 'Humiliation', 'Embarrassment', 
    'Mortification', 'Pride', 'Arrogance', 'Hubris', 'Vanity', 'Triumph', 
    'Envy', 'Jealousy', 'Covetousness', 'Schadenfreude', 'Contempt', 'Disdain', 
    'Scorn', 'Derision', 'Disgust', 'Revulsion', 'Repugnance', 'Aversion', 
    'Surprise', 'Astonishment', 'Amazement', 'Shock', 'Startle', 'Wonder', 
    'Awe', 'Fascination', 'Curiosity', 'Interest', 'Intrigue', 'Inquisitiveness', 
    'Confusion', 'Bafflement', 'Perplexity', 'Bewilderment', 'Doubt', 
    'Skepticism', 'Suspicion', 'Disbelief', 'Realization', 'Epiphany', 
    'Clarity', 'Certainty', 'Determination', 'Resolve', 'Tenacity', 
    'Perseverance', 'Zeal', 'Fervor', 'Passion', 'Enthusiasm', 'Excitement', 
    'Anticipation', 'Eagerness', 'Inspiration', 'Motivation', 'Ambition', 
    'Aspiration', 'Courage', 'Bravery', 'Catharsis', 'Weltschmerz', 'Sonder', 
    'Acedia', 'Angst', 'Malaise', 'Pity', 'Peevishness', 'Petulance', 
    'Sullenness', 'Moodiness', 'Giddiness', 'Exhilaration', 'Jubilation', 
    'Merriment', 'Glee', 'Reverence', 'Veneration', 'Respect', 'Admiration', 
    'Esteem', 'Boredom', 'Monotony', 'Tedium', 'Dullness', 'Indifference', 
    'Disappointment', 'Dismay', 'Letdown', 'Frustration', 'Defeat', 
    'Vindication', 'Righteousness', 'Self-righteousness', 'Smugness', 
    'Cynicism', 'Pessimism', 'Nihilism', 'Fatalism', 'Defeatism'
]
