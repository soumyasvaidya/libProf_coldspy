- Application Name: Chameleon
- Version: 4.1.0
- Lambda function repo: https://github.com/ddps-lab/serverless-faas-workbench/tree/master/aws/cpu-memory/chameleon
- Main Library Repo: https://github.com/malthe/chameleon

Optimization: Replaced pkg_resources with importlib-resources

### Average Initialization latency of
- Original code: 296.8836
- Optimized code: 254.2066911764706

#### Average Intialization latency reduced: 14.37%

### Average End to End latency of
- Original code: 534.81832
- Optimized code: 509.25963235294114

#### Average End to End latency reduced: 4.77%

### Average Memory Utilization of
- Original code: 48.208
- Optimized code: 44.27205882352941

#### Average Memory Utilization reduced: 8.16%

## 100+ Cold Starts
## Initialization:
![](init.png)

## Execution:
![](exec.png)

## End-to-end:
![](e2e.png)

## Optimization Diff
```diff
Binary files package_before/.DS_Store and package_after/.DS_Store differ
Binary files package_before/chameleon/.DS_Store and package_after/chameleon/.DS_Store differ
diff -x *.pyc -bur --co package_before/chameleon/compiler.py package_after/chameleon/compiler.py
--- package_before/chameleon/compiler.py	2024-03-17 23:27:28
+++ package_after/chameleon/compiler.py	2024-03-17 23:27:28
@@ -962,7 +962,7 @@
 
     global_builtins = set(builtins.__dict__)
 
-    def __init__(self, engine_factory, node, filename, source,
+    def __init__(self, engine_factory, node, spec, source,
                  builtins={}, strict=True):
         self._scopes = [set()]
         self._expression_cache = {}
@@ -1037,7 +1037,7 @@
                 self.lock.release()
 
         self.code = "\n".join((
-            "__filename = %r\n" % filename,
+            "__spec = %r\n" % spec,
             token_map_def,
             generator.code
         ))
@@ -1166,7 +1166,7 @@
 
         exc_handler = template(
             "if pos is not None: rcontext.setdefault('__error__', [])."
-            "append(token + (__filename, exc, ))",
+            "append(token + (__spec, exc, ))",
             exc=exc,
             token=template("__tokens[pos]", pos="__token", mode="eval"),
             pos="__token"
diff -x *.pyc -bur --co package_before/chameleon/loader.py package_after/chameleon/loader.py
--- package_before/chameleon/loader.py	2024-03-17 23:27:28
+++ package_after/chameleon/loader.py	2024-03-17 23:27:28
@@ -9,8 +9,6 @@
 from importlib.machinery import SourceFileLoader
 from threading import RLock
 
-import pkg_resources
-
 from .utils import encode_string
 
 
@@ -31,18 +29,6 @@
     return load
 
 
-def abspath_from_asset_spec(spec):
-    pname, filename = spec.split(':', 1)
-    return pkg_resources.resource_filename(pname, filename)
-
-
-if os.name == "nt":
-    def abspath_from_asset_spec(spec, f=abspath_from_asset_spec):
-        if spec[1] == ":":
-            return spec
-        return f(spec)
-
-
 class TemplateLoader:
     """Template loader class.
 
@@ -80,10 +66,7 @@
         if self.default_extension is not None and '.' not in spec:
             spec += self.default_extension
 
-        if ':' in spec:
-            spec = abspath_from_asset_spec(spec)
-
-        if not os.path.isabs(spec):
+        if ':' not in spec and not os.path.isabs(spec):
             for path in self.search_path:
                 path = os.path.join(path, spec)
                 if os.path.exists(path):
diff -x *.pyc -bur --co package_before/chameleon/program.py package_after/chameleon/program.py
--- package_before/chameleon/program.py	2024-03-17 23:27:28
+++ package_after/chameleon/program.py	2024-03-17 23:27:28
@@ -23,10 +23,10 @@
 
     restricted_namespace = True
 
-    def __init__(self, source, mode="xml", filename=None, tokenizer=None):
+    def __init__(self, source, mode="xml", spec=None, tokenizer=None):
         if tokenizer is None:
             tokenizer = self.tokenizers[mode]
-        tokens = tokenizer(source, filename)
+        tokens = tokenizer(source, spec)
         parser = ElementParser(
             tokens,
             self.DEFAULT_NAMESPACES,
diff -x *.pyc -bur --co package_before/chameleon/template.py package_after/chameleon/template.py
--- package_before/chameleon/template.py	2024-03-17 23:27:28
+++ package_after/chameleon/template.py	2024-03-17 23:27:28
@@ -5,6 +5,13 @@
 import sys
 import tempfile
 
+
+try:
+    # we need to try the backport first, as we rely on ``files`` added in 3.9
+    import importlib_resources
+except ImportError:
+    import importlib.resources as importlib_resources
+
 from .compiler import Compiler
 from .config import AUTO_RELOAD
 from .config import CACHE_DIRECTORY
@@ -19,10 +26,12 @@
 from .utils import DebuggingOutputStream
 from .utils import Scope
 from .utils import create_formatted_exception
+from .utils import detect_encoding
 from .utils import join
 from .utils import mangle
 from .utils import raise_with_traceback
 from .utils import read_bytes
+from .utils import read_xml_encoding
 from .utils import value_repr
 
 
@@ -32,25 +41,37 @@
     RecursionError = RuntimeError
 
 
+""" def get_package_versions():
+    try:
+        # try backport first as packages_distributions was added in Python 3.10
+        import importlib_metadata
+    except ImportError:
+        import importlib.metadata as importlib_metadata
+
+    versions = {
+        x: importlib_metadata.version(x)
+        for x in sum(importlib_metadata.packages_distributions().values(), [])}
+    return sorted(versions.items()) """
+
 def get_package_versions():
     try:
-        import pkg_resources
+        # Try importing importlib.metadata from importlib_metadata (Python 3.10+)
+        import importlib_metadata
+        distributions = importlib_metadata.distributions()
     except ImportError:
-        logging.info("Setuptools not installed. Unable to determine version.")
-        return []
+        # Fallback to importing importlib.metadata (Python < 3.10)
+        import importlib.metadata as importlib_metadata
+        distributions = importlib_metadata.distributions()
 
-    versions = dict()
-    for path in sys.path:
-        for distribution in pkg_resources.find_distributions(path):
-            if distribution.has_version():
-                versions.setdefault(
-                    distribution.project_name,
-                    distribution.version,
-                )
+    # Extract package versions from distributions
+    versions = {}
+    for distribution in distributions:
+        versions[distribution.metadata['Name']] = distribution.version
 
     return sorted(versions.items())
 
 
+
 pkg_digest = hashlib.sha1(__name__.encode('utf-8'))
 for name, version in get_package_versions():
     pkg_digest.update(name.encode('utf-8'))
@@ -87,11 +108,12 @@
     """
 
     default_encoding = "utf-8"
+    default_content_type = None
 
     # This attribute is strictly informational in this template class
     # and is used in exception formatting. It may be set on
-    # initialization using the optional ``filename`` keyword argument.
-    filename = '<string>'
+    # initialization using the optional ``spec`` keyword argument.
+    spec = '<string>'
 
     _cooked = False
 
@@ -143,7 +165,7 @@
         return self.render(**kwargs)
 
     def __repr__(self):
-        return "<{} {}>".format(self.__class__.__name__, self.filename)
+        return "<{} {}>".format(self.__class__.__name__, self.spec)
 
     @property
     def keep_body(self):
@@ -224,11 +246,15 @@
             body, encoding, content_type = read_bytes(
                 body, self.default_encoding
             )
+        elif body.startswith('<?xml'):
+            content_type = 'text/xml'
+            encoding = read_xml_encoding(body.encode("utf-8"))
         else:
-            content_type = body.startswith('<?xml')
-            encoding = None
+            content_type, encoding = detect_encoding(
+                body, self.default_encoding
+            )
 
-        self.content_type = content_type
+        self.content_type = content_type or self.default_content_type
         self.content_encoding = encoding
 
         self.cook(body)
@@ -244,12 +270,12 @@
                 source = self._compile(body, builtins)
                 if self.debug:
                     source = "# template: {}\n#\n{}".format(
-                        self.filename, source)
+                        self.spec, source)
                 if self.keep_source:
                     self.source = source
                 cooked = self.loader.build(source, filename)
             except TemplateError as exc:
-                exc.token.filename = self.filename
+                exc.token.filename = self.spec
                 raise
         elif self.keep_source:
             module = sys.modules.get(cooked.get('__name__'))
@@ -266,8 +292,8 @@
         sha.update(class_name)
         digest = sha.hexdigest()
 
-        if self.filename and self.filename is not BaseTemplate.filename:
-            digest = os.path.splitext(self.filename)[0] + '-' + digest
+        if self.spec and self.spec is not BaseTemplate.spec:
+            digest = os.path.splitext(self.spec.filename)[0] + '-' + digest
 
         return digest
 
@@ -275,12 +301,27 @@
         program = self.parse(body)
         module = Module("initialize", program)
         compiler = Compiler(
-            self.engine, module, self.filename, body,
+            self.engine, module, self.spec, body,
             builtins, strict=self.strict
         )
         return compiler.code
 
 
+class Spec(object):
+    __slots__ = ('filename', 'pname', 'spec')
+
+    def __init__(self, spec):
+        self.spec = spec
+        if ':' in spec:
+            (self.pname, self.filename) = spec.split(':', 1)
+        else:
+            self.pname = None
+            self.filename = spec
+
+    def __repr__(self):
+        return repr(self.spec)
+
+
 class BaseTemplateFile(BaseTemplate):
     """File-based template base class.
 
@@ -294,16 +335,17 @@
 
     def __init__(
             self,
-            filename,
+            spec,
             auto_reload=None,
             post_init_hook=None,
             **config):
+        if ':' not in spec:
         # Normalize filename
-        filename = os.path.abspath(
-            os.path.normpath(os.path.expanduser(filename))
+            spec = os.path.abspath(
+                os.path.normpath(os.path.expanduser(spec))
         )
 
-        self.filename = filename
+        self.spec = Spec(spec)
 
         # Override reload setting only if value is provided explicitly
         if auto_reload is not None:
@@ -327,44 +369,46 @@
 
         if self._cooked is False:
             body = self.read()
-            log.debug("cooking %r (%d bytes)..." % (self.filename, len(body)))
+            log.debug("cooking %r (%d bytes)..." % (self.spec, len(body)))
             self.cook(body)
 
     def mtime(self):
+        if self.spec.pname is not None:
+            return 0
+        else:
         try:
-            return os.path.getmtime(self.filename)
+                return os.path.getmtime(self.spec.filename)
         except OSError:
             return 0
 
     def read(self):
-        with open(self.filename, "rb") as f:
+        if self.spec.pname is not None:
+            files = importlib_resources.files(self.spec.pname)
+            data = files.joinpath(self.spec.filename).read_bytes()
+        else:
+            with open(self.spec.filename, "rb") as f:
             data = f.read()
 
         body, encoding, content_type = read_bytes(
             data, self.default_encoding
         )
 
-        # In non-XML mode, we support various platform-specific line
-        # endings and convert them to the UNIX newline character
-        if content_type != "text/xml" and '\r' in body:
-            body = body.replace('\r\n', '\n').replace('\r', '\n')
-
-        self.content_type = content_type
+        self.content_type = content_type or self.default_content_type
         self.content_encoding = encoding
 
         return body
 
     def _get_module_name(self, name):
-        filename = os.path.basename(self.filename)
+        filename = os.path.basename(self.spec.filename)
         mangled = mangle(filename)
         return "{}_{}.py".format(mangled, name)
 
-    def _get_filename(self):
-        return self.__dict__.get('filename')
+    def _get_spec(self):
+        return self.__dict__.get('spec')
 
-    def _set_filename(self, filename):
-        self.__dict__['filename'] = filename
+    def _set_spec(self, spec):
+        self.__dict__['spec'] = spec
         self._v_last_read = None
         self._cooked = False
 
-    filename = property(_get_filename, _set_filename)
+    spec = property(_get_spec, _set_spec)
\ No newline at end of file
diff -x *.pyc -bur --co package_before/chameleon/zpt/template.py package_after/chameleon/zpt/template.py
--- package_before/chameleon/zpt/template.py	2024-03-17 23:27:28
+++ package_after/chameleon/zpt/template.py	2024-03-17 23:27:28
@@ -27,6 +27,23 @@
     bytes = str
 
 
+BOOLEAN_HTML_ATTRIBUTES = [
+    # From http://www.w3.org/TR/xhtml1/#guidelines (C.10)
+    "compact",
+    "nowrap",
+    "ismap",
+    "declare",
+    "noshade",
+    "checked",
+    "disabled",
+    "readonly",
+    "multiple",
+    "selected",
+    "noresize",
+    "defer",
+]
+
+
 class PageTemplate(BaseTemplate):
     """Constructor for the page template language.
 
@@ -73,6 +90,10 @@
         The special return value ``default`` drops or inserts the
         attribute based on the value element attribute value.
 
+        The default setting is to autodetect if we're in HTML-mode and
+        provide the standard set of boolean attributes for this
+        document type.
+
       ``translate``
 
         Use this option to set a translation function.
@@ -171,12 +192,13 @@
     }
 
     default_expression = 'python'
+    default_content_type = 'text/html'
 
     translate = staticmethod(simple_translate)
 
     encoding = None
 
-    boolean_attributes = set()
+    boolean_attributes = None
 
     mode = "xml"
 
@@ -222,11 +244,22 @@
         return ExpressionParser(self.expression_types, self.default_expression)
 
     def parse(self, body):
+        boolean_attributes = self.boolean_attributes
+
+        if self.content_type != 'text/xml':
+            if boolean_attributes is None:
+                boolean_attributes = BOOLEAN_HTML_ATTRIBUTES
+
+            # In non-XML mode, we support various platform-specific
+            # line endings and convert them to the UNIX newline
+            # character.
+            body = body.replace('\r\n', '\n').replace('\r', '\n')
+
         return MacroProgram(
-            body, self.mode, self.filename,
+            body, self.mode, self.spec,
             escape=True if self.mode == "xml" else False,
             default_marker=self.default_marker,
-            boolean_attributes=self.boolean_attributes,
+            boolean_attributes=boolean_attributes or frozenset([]),
             implicit_i18n_translate=self.implicit_i18n_translate,
             implicit_i18n_attributes=self.implicit_i18n_attributes,
             trim_attribute_space=self.trim_attribute_space,
@@ -379,7 +412,7 @@
 
     prepend_relative_search_path = True
 
-    def __init__(self, filename, search_path=None, loader_class=TemplateLoader,
+    def __init__(self, spec, search_path=None, loader_class=TemplateLoader,
                  **config):
         if search_path is None:
             search_path = []
@@ -393,7 +426,7 @@
             # If the flag is set (this is the default), prepend the path
             # relative to the template file to the search path
             if self.prepend_relative_search_path:
-                path = dirname(self.filename)
+                path = dirname(self.spec.filename)
                 search_path.insert(0, path)
 
             loader = loader_class(search_path=search_path, **config)
@@ -404,7 +437,7 @@
             self._loader = loader.bind(template_class)
 
         super().__init__(
-            filename,
+            spec,
             post_init_hook=post_init,
             **config
         )

```