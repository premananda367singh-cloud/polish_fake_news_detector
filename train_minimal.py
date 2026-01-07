#!/usr/bin/env python3
"""
Minimal training script for Polish Fake News Detector
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("MINIMAL TRAINING FOR POLISH FAKE NEWS DETECTOR")
print("=" * 60)

# Sample Polish fake/real news data
print("\n1. Creating sample training data...")

# Real news examples (more formal, factual)
real_news_examples = [
    "Prezydent Andrzej Duda podpisał w piątek ustawę o wsparciu młodych naukowców.",
    "Ministerstwo Zdrowia poinformowało o spadku liczby zachorowań na grypę.",
    "Rząd przyjął projekt ustawy o zmianie przepisów podatkowych.",
    "Główny Urząd Statystyczny opublikował dane o inflacji za ostatni miesiąc.",
    "Minister edukacji ogłosił nowy program nauczania w szkołach.",
    "Bank Centralny podjął decyzję o utrzymaniu stóp procentowych.",
    "Polski rząd podpisał umowę handlową z Unią Europejską.",
    "Narodowy Fundusz Zdrowia zwiększył finansowanie badań naukowych.",
    "Prezydent Warszawy ogłosił plan rozwoju transportu publicznego.",
    "Ministerstwo Klimatu przedstawiło strategię energetyczną na lata 2024-2030."
]

# Fake news examples (sensational, emotional, conspiracy theories)
fake_news_examples = [
    "SZOK! Naukowcy odkryli, że szczepionki powodują AUTYZM!",
    "KONSPIRACJA! Rząd ukrywa prawdę o prawdziwych przyczynach pandemii!",
    "NIEUWAGO! Koncerny farmaceutyczne testują na nas niebezpieczne leki!",
    "TAJEMNICA! Elity ukrywają prawdziwy cel 5G!",
    "WSTRZĄSAJĄCE! Politycy przyjmują łapówki od zagranicznych korporacji!",
    "UKRYWANE! Prawdziwe dane o śmiertelności po szczepieniach!",
    "SPISEK! Media głównego nurtu manipulują opinią publiczną!",
    "NIEBEZPIECZEŃSTWO! Nowa technologia pozwala na kontrolę umysłów!",
    "ZDECYDUJ SAM! Czy wiesz, co naprawdę znajduje się w twojej żywności?",
    "ALARM! Władze planują wprowadzenie obowiązkowych szczepień dla wszystkich!"
]

# Combine and create labels
all_texts = real_news_examples + fake_news_examples
all_labels = [1] * len(real_news_examples) + [0] * len(fake_news_examples)

print(f"   Created {len(all_texts)} training examples:")
print(f"   - Real news: {len(real_news_examples)}")
print(f"   - Fake news: {len(fake_news_examples)}")

# Train TF-IDF model
print("\n2. Training TF-IDF model...")
try:
    from models.tfidf_detector import TfidfDetector
    
    tfidf_model = TfidfDetector()
    tfidf_model.train(all_texts, all_labels)
    print("   ✓ TF-IDF model trained successfully")
    
    # Test the trained model
    test_text = "Prezydent podpisał nową ustawę"
    result = tfidf_model.predict(test_text)
    print(f"   Test prediction: {'REAL' if result['prediction'] == 1 else 'FAKE'}")
    print(f"   Confidence: {result['confidence']:.2%}")
    
except Exception as e:
    print(f"   ✗ Error training TF-IDF: {e}")
    import traceback
    traceback.print_exc()

# Train ensemble (optional - might need transformers)
print("\n3. Training ensemble model (TF-IDF only mode)...")
try:
    from models.ensemble_detector import EnsembleDetector
    
    # Create ensemble with just TF-IDF for now
    ensemble = EnsembleDetector()
    
    # Train only TF-IDF part
    ensemble.models['tfidf'].train(all_texts, all_labels)
    ensemble.is_trained = True
    
    print("   ✓ Ensemble model (TF-IDF) trained")
    
    # Test ensemble
    test_text = "Szokujące odkrycie naukowców!"
    result = ensemble.predict(test_text)
    print(f"   Test prediction: {'REAL' if result['prediction'] == 1 else 'FAKE'}")
    print(f"   Confidence: {result['confidence']:.2%}")
    
    # Save the trained model
    print("\n4. Saving trained models...")
    os.makedirs('models/saved', exist_ok=True)
    
    # Save TF-IDF model
    import pickle
    with open('models/saved/tfidf_model.pkl', 'wb') as f:
        pickle.dump(tfidf_model, f)
    print("   ✓ TF-IDF model saved to models/saved/tfidf_model.pkl")
    
    # Save ensemble
    with open('models/saved/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)
    print("   ✓ Ensemble model saved to models/saved/ensemble_model.pkl")
    
except Exception as e:
    print(f"   ✗ Error with ensemble: {e}")
    # Continue anyway

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)
print("\nNow you can run the test with trained models.")
print("\nRun: python test_trained.py")