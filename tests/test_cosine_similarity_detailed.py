"""
دقیق‌ترین تست برای CosineSimilarity
تحت بررسی: آیا الگوریتم درست پیاده‌سازی شده است؟
"""

import pytest
import math
from graphion.core.models import FeatureSet
from graphion.builders.relation.cosine_similarity import CosineSimilarity


class TestCosineSimilarityCorrectness:
    """تست‌های تحقیق درستی الگوریتم"""
    
    def test_known_value_orthogonal_vectors(self):
        """
        تست 1: بردارهای متعامد (Orthogonal Vectors)
        
        دو بردار متعامد باید cosine similarity = 0 داشته باشند
        مثال: [1, 0] · [0, 1] = 0
        """
        feature_set = FeatureSet.from_lists(
            ids=["v1", "v2"],
            features=[[1.0, 0.0], [0.0, 1.0]],
        )
        
        builder = CosineSimilarity()
        relations = builder.build(feature_set)
        
        # بردارهای متعامد: cosine similarity = 0
        orthogonal_relations = [
            r for r in relations 
            if (r.source == "v1" and r.target == "v2") or 
               (r.source == "v2" and r.target == "v1")
        ]
        
        for relation in orthogonal_relations:
            assert abs(relation.weight) < 1e-10, \
                f"بردارهای متعامد باید similarity ≈ 0 داشته باشند، نه {relation.weight}"
    
    
    def test_known_value_identical_vectors(self):
        """
        تست 2: بردارهای یکسان (Identical Vectors)
        
        cosine similarity([v, v]) = 1 (کاملاً مشابه)
        """
        feature_set = FeatureSet.from_lists(
            ids=["v1", "v2"],
            features=[[3.0, 4.0], [3.0, 4.0]],
        )
        
        builder = CosineSimilarity()
        relations = builder.build(feature_set)
        
        # بردارهای یکسان
        identical_relations = [
            r for r in relations 
            if r.source == "v1" and r.target == "v2"
        ]
        
        for relation in identical_relations:
            assert abs(relation.weight - 1.0) < 1e-10, \
                f"بردارهای یکسان باید similarity = 1 داشته باشند، نه {relation.weight}"
    
    
    def test_known_value_opposite_vectors(self):
        """
        تست 3: بردارهای مخالف (Opposite Vectors)
        
        cosine similarity([-v, v]) = -1 (کاملاً مخالف)
        """
        feature_set = FeatureSet.from_lists(
            ids=["v1", "v2"],
            features=[[1.0, 0.0], [-1.0, 0.0]],
        )
        
        builder = CosineSimilarity()
        relations = builder.build(feature_set)
        
        opposite_relations = [
            r for r in relations 
            if r.source == "v1" and r.target == "v2"
        ]
        
        for relation in opposite_relations:
            assert abs(relation.weight - (-1.0)) < 1e-10, \
                f"بردارهای مخالف باید similarity = -1 داشته باشند، نه {relation.weight}"
    
    
    def test_known_value_manual_calculation(self):
        """
        تست 4: محاسبه دستی برای تأیید فرمول
        
        cosine(a, b) = (a · b) / (||a|| * ||b||)
        
        a = [3, 4]
        b = [1, 0]
        
        a · b = 3*1 + 4*0 = 3
        ||a|| = sqrt(9 + 16) = 5
        ||b|| = sqrt(1 + 0) = 1
        cosine = 3 / 5 = 0.6
        """
        feature_set = FeatureSet.from_lists(
            ids=["a", "b"],
            features=[[3.0, 4.0], [1.0, 0.0]],
        )
        
        builder = CosineSimilarity()
        relations = builder.build(feature_set)
        
        ab_relation = [r for r in relations if r.source == "a" and r.target == "b"][0]
        
        expected = 0.6
        assert abs(ab_relation.weight - expected) < 1e-10, \
            f"محاسبه دستی: انتظار {expected}، اما {ab_relation.weight} دریافت شد"
    
    
    def test_symmetry_property(self):
        """
        تست 5: خاصیت تقارن (Symmetry)
        
        cosine(a, b) = cosine(b, a)
        """
        feature_set = FeatureSet.from_lists(
            ids=["x", "y"],
            features=[[2.0, 3.0], [4.0, 5.0]],
        )
        
        builder = CosineSimilarity()
        relations = builder.build(feature_set)
        
        xy_relation = [r for r in relations if r.source == "x" and r.target == "y"][0]
        yx_relation = [r for r in relations if r.source == "y" and r.target == "x"][0]
        
        assert abs(xy_relation.weight - yx_relation.weight) < 1e-10, \
            f"تقارن نقض شده: cosine(x,y) = {xy_relation.weight} ≠ cosine(y,x) = {yx_relation.weight}"
    
    
    def test_zero_vector_handling(self):
        """
        تست 6: بردار صفر (Zero Vector)
        
        cosine(zero_vector, any_vector) = 0 (تعریف استاندارد)
        """
        feature_set = FeatureSet.from_lists(
            ids=["zero", "nonzero"],
            features=[[0.0, 0.0], [3.0, 4.0]],
        )
        
        builder = CosineSimilarity()
        relations = builder.build(feature_set)
        
        zero_relations = [
            r for r in relations 
            if r.source == "zero" and r.target == "nonzero"
        ]
        
        for relation in zero_relations:
            assert relation.weight == 0.0, \
                f"بردار صفر باید similarity = 0 داشته باشند، نه {relation.weight}"
    
    
    def test_affinity_conversion_negative_to_zero(self):
        """
        تست 7: تبدیل Affinity
        
        affinity(raw_score) = max(0, raw_score)
        برای bردارهای مخالف: cosine = -1 → affinity = 0
        """
        feature_set = FeatureSet.from_lists(
            ids=["a", "b"],
            features=[[1.0], [-1.0]],
        )
        
        builder = CosineSimilarity()
        raw_score = builder.score([1.0], [-1.0])
        affinity = builder.affinity(raw_score)
        
        assert raw_score == -1.0, \
            f"Raw score برای بردارهای مخالف باید -1 باشد، نه {raw_score}"
        assert affinity == 0.0, \
            f"Affinity برای منفی باید 0 باشد، نه {affinity}"
    
    
    def test_normalized_vectors(self):
        """
        تست 8: بردارهای نرمالایز‌شده
        
        اگر normalize=True در BaseNumericRelationBuilder
        بردارها باید به L2 norm = 1 تبدیل شوند
        
        برای بردارهای نرمالایز‌شده:
        cosine(u, v) = u · v
        """
        feature_set = FeatureSet.from_lists(
            ids=["p", "q"],
            features=[[3.0, 4.0], [5.0, 12.0]],
        )
        
        # با normalize=False (پیش‌فرض)
        builder_unnormalized = CosineSimilarity(normalize=False)
        relations_unnormalized = builder_unnormalized.build(feature_set)
        
        pq_relation = [r for r in relations_unnormalized 
                      if r.source == "p" and r.target == "q"][0]
        
        # محاسبه دستی
        # p = [3, 4], q = [5, 12]
        # p · q = 15 + 48 = 63
        # ||p|| = 5, ||q|| = 13
        # cosine = 63 / 65 ≈ 0.969
        expected = 63 / 65
        assert abs(pq_relation.weight - expected) < 1e-10, \
            f"انتظار {expected}، اما {pq_relation.weight} دریافت شد"


class TestEdgeCases:
    """تست‌های حالات خاص"""
    
    def test_high_dimensional_vectors(self):
        """تست بردارهای بعد بالا (1000 بعد)"""
        feature_set = FeatureSet.from_lists(
            ids=["v1", "v2"],
            features=[
                [float(i) for i in range(1000)],
                [float(i+1) for i in range(1000)],
            ],
        )
        
        builder = CosineSimilarity()
        relations = builder.build(feature_set)
        
        # باید similarity در محدوده [-1, 1] باشد
        for relation in relations:
            assert -1.0 <= relation.weight <= 1.0, \
                f"Similarity باید در [-1, 1] باشد، نه {relation.weight}"
    
    
    def test_small_magnitude_values(self):
        """تست مقادیر بسیار کوچک"""
        feature_set = FeatureSet.from_lists(
            ids=["tiny1", "tiny2"],
            features=[[1e-10, 2e-10], [2e-10, 4e-10]],
        )
        
        builder = CosineSimilarity()
        relations = builder.build(feature_set)
        
        # باید درست محاسبه شود حتی برای اعداد کوچک
        relation = [r for r in relations 
                   if r.source == "tiny1" and r.target == "tiny2"][0]
        
        # یکسان هستند (scaled versions)
        assert abs(relation.weight - 1.0) < 1e-8, \
            f"بردارهای proportional باید similarity ≈ 1 داشته باشند"


class TestConsistency:
    """تست‌های سازگاری"""
    
    def test_consistency_across_calls(self):
        """نتایج باید مستقل از نوبت‌های متعدد یکسان باشد"""
        feature_set = FeatureSet.from_lists(
            ids=["a", "b", "c"],
            features=[[1, 2], [3, 4], [5, 6]],
        )
        
        builder = CosineSimilarity()
        
        result1 = builder.build(feature_set)
        result2 = builder.build(feature_set)
        
        for r1, r2 in zip(result1.relations, result2.relations):
            assert r1.weight == r2.weight, \
                "نتایج باید deterministic باشند"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
