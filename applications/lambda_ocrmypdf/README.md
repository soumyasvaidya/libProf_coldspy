- Application Name: OCRmyPDF
- Version: 14.4.0
- Repo: https://github.com/ocrmypdf/OCRmyPDF

Optimization: Removed rich.console because it is not needed, lazy loaded pdfminer

1. `./force_cold_start.sh` will run 100 cold starts in concurrently

## 100+ Cold Starts
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)

### Average Initialization latency of
- Original code: 1297.4333333333334
- Optimized code: 914.1660402684565

#### Average Intialization latency reduced: 29.54%

### Average End to End latency of
- Original code: 5542.114444444444
- Optimized code: 4667.022483221475

#### Average End to End latency reduced: 15.78%

### Average Memory Utilization of
- Original code: 210.55555555555554
- Optimized code: 193.6778523489933

#### Average Memory Utilization reduced: 8.01%

Optimization diff
```diff
diff -x *.pyc -bur --co python/ocrmypdf/_sync.py python_after/ocrmypdf/_sync.py
--- python/ocrmypdf/_sync.py	2024-03-17 22:44:58
+++ python_after/ocrmypdf/_sync.py	2024-03-17 00:36:36
@@ -25,7 +25,7 @@
 from ocrmypdf._concurrent import Executor, setup_executor
 from ocrmypdf._graft import OcrGrafter
 from ocrmypdf._jobcontext import PageContext, PdfContext, cleanup_working_files
-from ocrmypdf._logging import PageNumberFilter
+# from ocrmypdf._logging import PageNumberFilter
 from ocrmypdf._pipeline import (
     convert_to_pdfa,
     copy_final,
@@ -327,7 +327,7 @@
         '[%(asctime)s] - %(name)s - %(levelname)7s -%(pageno)s %(message)s'
     )
     log_file_handler.setFormatter(formatter)
-    log_file_handler.addFilter(PageNumberFilter())
+    # log_file_handler.addFilter(PageNumberFilter())
     logging.getLogger(prefix).addHandler(log_file_handler)
     return log_file_handler
 
diff -x *.pyc -bur --co python/ocrmypdf/api.py python_after/ocrmypdf/api.py
--- python/ocrmypdf/api.py	2024-03-17 22:44:58
+++ python_after/ocrmypdf/api.py	2024-03-17 00:36:59
@@ -15,7 +15,7 @@
 from typing import AnyStr, BinaryIO, Iterable, Union
 from warnings import warn
 
-from ocrmypdf._logging import PageNumberFilter, TqdmConsole
+# from ocrmypdf._logging import PageNumberFilter, TqdmConsole
 from ocrmypdf._plugin_manager import get_plugin_manager
 from ocrmypdf._sync import run_pipeline
 from ocrmypdf._validation import check_options
@@ -89,8 +89,8 @@
     log.setLevel(logging.DEBUG)
 
     console = None
-    if plugin_manager and progress_bar_friendly:
-        console = plugin_manager.hook.get_logging_console()
+    # if plugin_manager and progress_bar_friendly:
+    #     console = plugin_manager.hook.get_logging_console()
 
     if not console:
         console = logging.StreamHandler(stream=sys.stderr)
@@ -102,7 +102,7 @@
     else:
         console.setLevel(logging.INFO)
 
-    console.addFilter(PageNumberFilter())
+    # console.addFilter(PageNumberFilter())
 
     if verbosity >= 2:
         fmt = '%(levelname)7s %(name)s -%(pageno)s %(message)s'
@@ -338,8 +338,8 @@
 
 
 __all__ = [
-    'PageNumberFilter',
-    'TqdmConsole',
+    # 'PageNumberFilter',
+    # 'TqdmConsole',
     'Verbosity',
     'check_options',
     'configure_logging',
diff -x *.pyc -bur --co python/ocrmypdf/builtin_plugins/concurrency.py python_after/ocrmypdf/builtin_plugins/concurrency.py
--- python/ocrmypdf/builtin_plugins/concurrency.py	2024-03-17 22:44:58
+++ python_after/ocrmypdf/builtin_plugins/concurrency.py	2024-03-17 22:37:29
@@ -16,10 +16,10 @@
 from contextlib import suppress
 from typing import Callable, Iterable, Type, Union
 
-from rich.console import Console as RichConsole
+# from rich.console import Console as RichConsole
 
 from ocrmypdf import Executor, hookimpl
-from ocrmypdf._logging import RichLoggingHandler, RichTqdmProgressAdapter
+# from ocrmypdf._logging import RichLoggingHandler, RichTqdmProgressAdapter
 from ocrmypdf.exceptions import InputFileError
 from ocrmypdf.helpers import remove_all_log_handlers
 
@@ -168,20 +168,20 @@
     return StandardExecutor(pbar_class=progressbar_class)
 
 
-RICH_CONSOLE = RichConsole(stderr=True)
+# RICH_CONSOLE = RichConsole(stderr=True)
 
 
-@hookimpl
-def get_progressbar_class():
-    """Return the default progress bar class."""
+# @hookimpl
+# def get_progressbar_class():
+#     """Return the default progress bar class."""
 
-    def partial_RichTqdmProgressAdapter(*args, **kwargs):
-        return RichTqdmProgressAdapter(*args, **kwargs, console=RICH_CONSOLE)
+#     def partial_RichTqdmProgressAdapter(*args, **kwargs):
+#         return RichTqdmProgressAdapter(*args, **kwargs, console=RICH_CONSOLE)
 
-    return partial_RichTqdmProgressAdapter
+#     return partial_RichTqdmProgressAdapter
 
 
-@hookimpl
-def get_logging_console():
-    """Return the default logging console handler."""
-    return RichLoggingHandler(console=RICH_CONSOLE)
+# @hookimpl
+# def get_logging_console():
+#     """Return the default logging console handler."""
+#     return RichLoggingHandler(console=RICH_CONSOLE)
diff -x *.pyc -bur --co python/ocrmypdf/pdfinfo/info.py python_after/ocrmypdf/pdfinfo/info.py
--- python/ocrmypdf/pdfinfo/info.py	2024-03-17 22:44:58
+++ python_after/ocrmypdf/pdfinfo/info.py	2024-03-17 19:55:33
@@ -35,7 +35,6 @@
 from ocrmypdf._concurrent import Executor, SerialExecutor
 from ocrmypdf.exceptions import EncryptedPdfError, InputFileError
 from ocrmypdf.helpers import Resolution, available_cpu_count, pikepdf_enable_mmap
-from ocrmypdf.pdfinfo.layout import get_page_analysis, get_text_boxes
 
 logger = logging.getLogger()
 
@@ -783,7 +782,10 @@
 
         check_this_page = pageno in check_pages
 
         if check_this_page and detailed_analysis:
+            from ocrmypdf.pdfinfo.layout import get_page_analysis, get_text_boxes
             pscript5_mode = str(pdf.docinfo.get(Name.Creator)).startswith('PScript5')
             miner = get_page_analysis(infile, pageno, pscript5_mode)
             self._textboxes = list(simplify_textboxes(miner, get_text_boxes))

```