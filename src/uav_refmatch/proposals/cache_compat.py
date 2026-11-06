"""Instance-local legacy cache bridge for the pinned Microsoft Florence code.

Transformers 4.57 otherwise pre-allocates a DynamicCache, whereas Microsoft's
decoder indexes tuple caches. Keep upstream files and other model classes intact.
Remove/revalidate this bridge when upgrading Transformers or the checkpoint code.
"""

from types import MethodType
import transformers


def _legacy_only(self):
    return False


def enable_legacy_cache(model):
    language_model = model.language_model
    if transformers.__version__ != "4.57.6":
        raise RuntimeError("Florence legacy cache bridge requires validated Transformers 4.57.6")
    if type(language_model).__name__ != "Florence2LanguageForConditionalGeneration":
        raise RuntimeError("Unexpected Florence language model; legacy cache bridge not applied")
    language_model._supports_default_dynamic_cache = MethodType(_legacy_only, language_model)
