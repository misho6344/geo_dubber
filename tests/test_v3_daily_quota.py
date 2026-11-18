"""When Gemini's daily quota is used up, the translator must stop waiting and leave the lines for later."""
from core import translation as T


class _Models:
    def __init__(self):
        self.calls = []

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 45000.0s")


def test_daily_quota_is_not_waited_on_and_lines_stay_empty():
    waits = []
    t = T.GeminiTranslator(model="gemini-3.5-flash-lite", api_key="k", batch=2, sleep=waits.append,
                           web_fallback=False)
    models = _Models()
    t.client = type("C", (), {"models": models})()
    out = t.translate(["hello", "there", "again"], "en", max_seconds_list=[5, 5, 5])
    assert out == ["", "", ""]
    assert sum(waits) == 0                       # no 120-second rounds
    assert len(models.calls) == 1                 # a hard daily quota must not probe other models
    assert t.stats.get("daily_quota") is True


def test_short_quota_delay_still_waits_and_retries():
    waits = []
    t = T.GeminiTranslator(model="gemini-3.5-flash-lite", api_key="k", batch=2, sleep=waits.append,
                           web_fallback=False)

    class _Short:
        def __init__(self):
            self.n = 0

        def generate_content(self, model, contents, config=None):
            self.n += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 2.0s")

    t.client = type("C", (), {"models": _Short()})()
    t.max_rounds = 2
    t.translate(["hello"], "en", max_seconds_list=[5])
    assert sum(waits) > 0                         # a short burst is still waited out



def test_billing_quota_stops_all_gemini_calls_for_the_rest_of_the_run():
    class BillingQuota:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config=None):
            self.calls.append(model)
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED: You exceeded your current quota, "
                "please check your plan and billing details."
            )

    models = BillingQuota()
    t = T.GeminiTranslator(model="gemini-3.5-flash-lite", api_key="k", batch=1,
                           web_fallback=False)
    t.client = type("C", (), {"models": models})()
    out = t.translate(["one", "two", "three"], "en", max_seconds_list=[5, 5, 5])

    assert out == ["", "", ""]
    assert len(models.calls) == 1
    assert t.stats["daily_quota"] is True
    assert t.stats["requests"] == 1
