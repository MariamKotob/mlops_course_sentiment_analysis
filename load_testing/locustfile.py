import random
from locust import HttpUser, task, between

# Sample Arabic sentences designed to match the typical length of tweets (~100-140 chars).
# Using representative sequence lengths ensures the transformer model takes realistic
# processing time during load testing, preventing bias in the benchmark results.
SAMPLE_TEXTS = [
    "هذا المنتج رائع جداً وأنا سعيد به لأن الجودة ممتازة وتستحق كل قرش دفعته فيه، أنصح الجميع بتجربته حقاً 👍",
    "خدمة العملاء كانت سيئة للغاية وتأخروا في الرد على استفساراتي ولم يحلوا المشكلة حتى الآن، لن أتعامل معكم مرة أخرى 😡",
    "التطبيق بطيء جداً ويحتاج إلى تحديث عاجل، الواجهة غير مريحة للمستخدم وتوجد الكثير من الأخطاء عند محاولة الدفع",
    "شكرا لكم على هذه الخدمة الرائعة والتوصيل السريع، لقد وصلت الشحنة في الوقت المحدد وكان التغليف ممتازاً جداً 😊",
    "بصراحة التجربة كانت متوسطة، هناك بعض الإيجابيات ولكن السلبيات تطغى عليها، أتمنى أن تأخذوا ملاحظاتي بعين الاعتبار",
    "لا أنصح أبداً بشراء هذا الهاتف، البطارية تنفد بسرعة كبيرة والكاميرا ليست بالجودة التي تم الإعلان عنها في الفيديو",
    "تجربة تسوق ممتازة من البداية للنهاية، الموقع سهل الاستخدام والأسعار معقولة جداً مقارنة بالمتاجر الأخرى في السوق",
    "لقد قمت بطلب المنتج منذ أكثر من أسبوعين ولم يصلني حتى الآن، ولا يوجد أي تجاوب من الدعم الفني، تجربة محبطة جداً",
]

class SentimentAnalysisUser(HttpUser):
    # Wait between 1 and 3 seconds between tasks
    wait_time = between(1, 3)

    @task(1)
    def health_check(self):
        """Check if the API and model are healthy"""
        self.client.get("/health", name="/health")

    @task(3)
    def predict_single(self):
        """Simulate a user predicting a single text"""
        text = random.choice(SAMPLE_TEXTS)
        self.client.post(
            "/predict",
            json={"text": text},
            name="/predict (single)"
        )

    @task(2)
    def predict_batch(self):
        """Simulate a batch prediction request"""
        # Random batch size between 2 and 5 (API max is MAX_BATCH_SIZE=64)
        batch_size = random.randint(2, 5)
        texts = random.choices(SAMPLE_TEXTS, k=batch_size)
        self.client.post(
            "/predict/batch",
            json={"texts": texts},
            name="/predict/batch"
        )

