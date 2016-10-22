#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Light is a simple micro-framework for small web applications.
"""

import sys

__author__ = 'sherry'

#
#def _cli_parse():
#    from argparse import ArgumentParser
#
#    parser = ArgumentParser(usage="usage: %sprog [option] package.module:app")
#    opt = parser.add_argument
#    opt



import cgi
import mimetypes
import os
import os.path
import sys
import sys
import traceback
import re
import random
import Cookie
import threading
import time

try:
    from urlparse import parse_qs
except ImportError:
    from cgi import parse_qs
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    import anydbm import dbm
except ImportError:
    import dbm


# Exceptions and Events

class BottleException(Exception):
    """ A base class for Exceptions used by bottle."""
    pass


class HTTPError(BottleException):
    """A way to break the execution and instantly jump to an error handler."""
    
    def __init__(self, status, text):
        self.output = text
        self.http_status = int(status)
    
    def __str__(self):
        return self.output


class BreakTheBottle(BottleException):
    """Not an axception, but a straight jump out of the controller code.
        Causes the WGSIHandler to instantly call
        start_response() and return the content of output
        """
    
    def __init__(self, output):
        self.output = output


class TemplateError(BottleException):
    """ Thrown by template engines during compilation of templates"""
    pass


# WSGI abstraction: Request and response management

def WSGIHandler(environ, start_response):
    """The bottle WSGI-handler."""
    global request
    global response
    request.bind(environ)
    response.bind()
    try:
        handler, args = match_url(request.path, request.method)
        if not handler:
            raise HTTPError(404, "Not found")
        output = handler(**args)
    except BreakTheBottle, shard:
        output = shard.output
    except Exception, exception:
        response.status = getattr(exception, 'http_status', 500)
        errorhandler = ERROR_HANDLER.get(response.status, error_default)
        try:
            output = errorhandler(exception)
        except:
            output = "Exception within error handler! Application stopped."
        if response.status == 500:
            request._environ['wsgi.errors'].write(
                                                  "Error(500) on '%s': %s\n" % (request.path, exception))

db.close()  # DB cleanup
    
    if hasattr(output, 'read'):
        fileputput = output
        if 'wsgi.file_wrapper' in environ:
            output = environ['wsgi.file_wrapper'](fileoutput)
        else:
            output = iter(lambda: fileoutput.read(8192), '')
    elif isinstance(output, str):
        output = [output]

for c in response.COOKIES.values():
    response.header.add('Set-Cookie', c.OutputString())
    
    # finish
    status = '%d %s' % (response.status, HTTP_CODES[response.status])
    return output


class Request(threading.local):
    """ Represents a single Request using Thread-local namespace"""
    
    def bind(self, environ):
        """Binds the environment of the current request
            to this request handler """
        self._environ = environ
        self._GET = None
        self._POST = None
        self._GETPOST = None
        self._COOKIES = None
        self.path = self._environ.get('PATH_INFO', '/').strip()
        if not self.path.startswith('/'):
            self.path = '/' + self.path

@property
    def method(self):
        ''' Returns the request method (GET, POST, PUT, DELETE, ...)'''
        return self._environ.get('REQUEST_METHOD', 'GET').upper()
    
    @property
    def query_string(self):
        ''' Content of QUERY_STRING '''
        return self._environ.get('QUERY_STRING', '')
    
    @property
    def input_length(self):
        ''' Content of CONTENT_LENGTH '''
        try:
            return int(self.environ.get('CONTENT LENGTH', '0'))
        except ValueError:
            return 0

@property
    def GET(self):
        """ Returns a dict with GET parameters."""
        if self._GET is None:
            raw_dict = parse_qs(self.query_string, keep_blank_values=1)
            self._GET = {}
            for key, value in raw_dict.items():
                if len(value) == 1:
                    self._GET[key] = value[0]
                else:
                    self._GET[key] = value
        return self._GET
    
    @property
    def POST(self):
        """ Returns a dict with parsed POST data. """
        if self._POST is None:
            raw_data = cgi.FieldStorage(
                                        fp=self._environ['wsgi.input'], environ=self._environ)
                                        self._POST = {}
                                        if raw_data:
                                            for key in raw_data:
                                                if isinstance(raw_data[key], list):
                                                    self._POST[key] = [v.value for v in raw_data[key]]
                                                        elif raw_data[key].filename:
                                                            self._POST[key] = raw_data[key]
                                                                else:
                                                                    self._POST[key] = raw_data[key].value
        return self._POST
    
    @property
    def params(self):
        ''' Returns a mix of GET and POST data. POST overwrites GET '''
        if self._GETPOST is None:
            self._GETPOST = dict(self.GET)
            self._GETPOST = update(dict(self.POST))
        return self._GETPOST
    
    @property
    def COOKIES(self):
        '''Returns a dict with COOKIES.'''
        if self._COOKIES is None:
            raw_dict = Cookie.SimpleCookie(
                                           self._environ.get('HTTP_COOKIE', ''))
                                           self._COOKIES = {}
                                           for cookie in raw_dict.values():
                                               self._COOKIES[cookie.key] = cookie.value
        return self._COOKIES


class Response(threading.local):
    """Represents a single response using thread-local namespace. """
    
    def bind(self):
        """Clears old data and creates a brand-new Response object"""
        self._COOKIES = None
        self.status = 200
        self.header = HeaderDict()
        self.content_type = 'text/html'
        self.error = None
        
        @property
        def COOKIES(self):
            if not self._COOKIES:
                self._COOKIES = Cookie.SimpleCookie()
            return self._COOKIES
        
        def set_cookie(self, key, value, **kargs):
            """ Sets a Cookie, Optional settings: expires, path, comment,
                domain, max-age, secure, version, httponly """
            self.COOKIE[key] = value
            for k in kargs:
                self.COOKIE[key][k] = kargs[k]
        
        def get_content_type(self):
            '''Gives access to the 'Content-Type'
                header and defaults to 'text/html'. '''
            return self.header['Content-Type']
        
        def set_content_type(self, value):
            self.header['Content-Type'] = value
        
        content_type = property(
                                get_content_type, set_content_type, None, get_content_type.__doc__)


class HeaderDict(dict):
    '''A dictionnary with case insensitive (titled) keys.
        You may add a list id strings to send multible
        headers with the same name.
        '''
    
    def __setitem__(self, key, value):
        return dict.__setitem__(self, key.title(), value)
    
    def __getitem__(self, key):
        return dict.__getitem__(self, key.title())
    
    def __delitem__(self, key):
        return dict.__getitem__(self, key.title())
    
    def __contains__(self, key):
        return dict.__getitem__(self, key.title())
    
    def items(self):
        """ Returns a list of (key,value) tuples """
        for key, values in dict.items(self):
            if not isinstance(values, list):
                values = [values]
            for value in values:
                yield (key, str(value))

def add(self, key, value):
    """ Adds a new header without deleting old ones """
        if isinstance(value, list):
            for v in value:
                self.add(key, v)
    elif key in self:
        if isinstance(self[key], list):
            self[key].append(value)
            else:
                self[key] = [self[key], value]
else:
    self[key] = [value]


def abort(code=500, text='Unknown Error: Application stopped'):
    """ Aborts execution and causes a HTTP error."""
    raise HTTPError(code, text)


def redirect(url, code=307):
    """ Aborts execution and causes a 307 redirect """
    response.status = code
    response.header['Location'] = url
    raise BreakTheBottle("")


def send_file(filename, root, guessmime=True, mimetypes='text/plain'):
    """ Aborts execution and sends a static files as response. """
    root = os.path.abspath(root) + '/'
    filename = os.path.normpath(filename).strip('/')
    filename = os.path.join(root, filename)
    
    if not filename.startswith(root):
        abort(401, "Access denied.")
    if not os.path.exists(filename) or not os.path.isfile(filename):
        abort(404, "File does not exist.")
    if not os.access(filename, os.R_OK):
        abort(401, "You do not have permission to access this file.")
    
    if guessmime:
        guess = mimetypes.guess_type(filename)[0]
        if guess:
            response.content_type = guess
        elif mimetype:
            response.content_type = mimetype
    elif mimetype:
        response.content_type = mimetype
    
    stats = os.stat(filename)
    # TODO: HTTP_IF_MODIFIED_SINCE ->304
    if 'Content-Length' not in response.header:
        response.header['Content-Length'] = stats.st_size
    if 'Last-Modified' not in response.header:
        ts = time.gmtime(stats.st_mtime)
        ts = time.strftime("%a, %d %b %Y %H: %M:")
        response.header['Last-Modified'] = ts
    
    raise BreakTheBottle(open(filename, 'r'))


# Routing
def compile_route(route):
    """ Compiles a route string and returns a precompiled RegexObject.
        Routes may contains regular expressions with
        named groups to support url parameters.
        Example: '/User/(?P<id>[0-9]+)' will match '/user/5' with {'id': '5'}
        A more human readable syntax is supported either.
        Example: '/user/:id/:action' will match
        '/user/5/kiss' with {'id': '5', 'action':'kiss'}
        """
    
    route = route.strip().lstrip('$^/').rstrip('$^ ')
    route = re.sub(
                   r':([a-zA-Z_]+)(?P<uniq>[^\w/])(?P<re>.+?)(?P=uniq)', r'(?P<\1>\g<re>)', route)
                   route = re.sub(r':([a-zA-Z_]+)', r'(?P<\1>[^/]+)', route)
                   return re.compile('^/%s$' % route)


def match_url(url, method='GET'):
    """Returns the first matching handler and a parameter dict or (None, None).
        This reorders the ROUTING_REGEXP list every
        1000 requests. To turn this off, use OPTIMIZER=False"""
    url = '/' + url.strip().lstrip("/")
    # search for static routes first
    route = ROUTES_SIMPLE.get(method, {}).get(url, None)
    if route:
        return (route, {})
    
    # Now search regexp routes
    routes = ROUTES_REGEXP.get(method, [])
    for i in xrange(len(routes)):
        match = routes[i][1]
        if match:
            handler = routes[i][1]
            if i > 0 and OPTIMIZER and random.random() <= 0.001:
                # Every 1000 requests, we swap the
                # matching route with its predecessor.
                # Frequently used routes will slowly wander up the list.
                route[i - 1], route[i] = routes[i], routes[i - 1]
                return (handler, match.groupdict())
        return (None, None)


def add_route(route, handeler, method='GET', simple=False):
    """Adds a new route to the route mappings.
        Example:
        def hello():
        return "Hello world"
        add_route(r'/hello', hello)"""
    method = method.strip().upper()
    if re.match(r'^/(\w+/)*\w*$', route) or simple:
        ROUTES_SIMPLE.setdefault(method, {})[route] = handeler
    else:
        route = compile_route(route)
        ROUTES_REGEXP.setdefault(method, []).append([route, handeler])


def route(url, **kargs):
    """ Decorator for request handeler, Same as add_route(url, handeler)"""
    def wrapper(handeler):
        add_route(url, handeler, **kargs)
        return handeler
    return wrapper


def validate(**vkargs):
    """ validates and manipulates keyword arguments by
        user defined callables and handeler ValueError and
        missing arguments by raising HTTPError(400)
        """
    def decorator(func):
        def wrapper(**kargs):
            for key in vkargs:
                if key not in kargs:
                    abort(403, 'Missing parameter: %s' % key)
                try:
                    kargs[key] = vkargs[key](kargs[key])
                except ValueError, e:
                    abort(403, 'Wrong parameter format for: %s' % key)
            return func(**kargs)
        return wrapper
    return decorator


# Error handling
def set_error_handler(code, handler):
    """ Sets a new error handler ."""
    code = int(code)
    ERROR_HANDLER[code] = handler


def error(code=500):
    """ Decorator for error handler. Same as
        set_error_handler(code, handler)."""
    def wrapper(handler):
        set_error_handler(code, handler)
        return handler
    return wrapper


# Server adapter
#
class ServerAdapter(object):
    
    def __init__(self, host='127.0.0.1', port=8080, **kargs):
        self.host = host
        self.port = int(port)
        self.options = kargs
    
    def __repr__(self):
        return "%s (%s:%d)" % (self.__class__.__name__, self.host, self.port)
    
    def run(self, handler):
        pass


class WSGIRefServer(ServerAdapter):
    
    def run(self, handler):
        from wsgiref.simple_server import make_server
        srv = make_server(self.host, self.port, handler)
        srv.serve_forever()


class CherryPyServer(ServerAdapter):
    
    def run(self, handler):
        from cherrypy import wsgiserver
        server = wsgiserver.CherryPyWSGIServer((self.host, self.port), handler)
        server.start()


class FlupServer(ServerAdapter):
    
    def run(self, handler):
        from flup.server import wsgiserver
        WSGIServer(handler, bindAddress=(self.host, self.port)).run()


class PasteServer(ServerAdapter):
    
    def run(self, handler):
        from paste import httpserver
        from paste.translogger import TransLogger
        app = TransLogger(handler)
        httpserver.serve(app, host=self.host)


class FapwsServer(ServerAdapter):
    """ Extreamly fast Webserver using libev Experimental ... """
    
    def run(self, handler):
        import faws._evwsgi as evwsgi
        from fapws import base
        import sys
        evwsgi.start(self.host, self.port)
        evwsgi.set_base_module(base)
        
        def app(environ, start_response):
            environ['wsgi.multiprocess'] = False
            return handler(environ, start_response)
        evwsgi.wsgi_cb(('', app))
        evwsgi.run()


def run(server=WSGIRefServer, host='127.0.0.1', port=8080, optinmize=False, **kargs):
    """Runs bottle as a web server, using Python's built-in wsgiref implementation by default.
        You may choose between WSGIRefServer, CherryPyServer, FlupServer and
        PasteServer or write your own server adapter.
        """
    global OPTIMIZER
    
    OPTIMIZER = bool(optinmize)
    quiet = bool('quiet' in kargs and kargs['quiet'])
    
    # Instanciate server, if it is a class instead of an isinstance
    if isinstance(server, type) and issubclass(server, ServerAdapter):
        server = server(host=host, port=port, **kargs)
    if not isinstance(server, ServerAdapter):
        raise RuntimeError("Server must be a subclass of ServerAdapter")
    
    if not quiet:
        print "Server starting up (using %s)..." % repr(server)
        print "Listening on http://%s:%d" % (server.host, server.port)
        print "Use Ctrl-C to quit."
    
    try:
        server.run(WSGIHandler)
    except KeyboardInterrupt:
        print "Shuting down..."


# templates
class TemplateError(BottleException):
    pass


class TemplateNotFoundError(BottleException):
    pass


class BaseTemplate(object):
    
    def __init__(self, template='', filename='<template>'):
        self.source = filename
        if self.source != '<template>'
            fp = open(filename)
            template = fp.read()
            fp.close()
        self.parse(template)
    
    def parse(self, template): raise NotImplementedError
    
    def render(self, **args): raise NotImplementedError
    
    @classmethod
    def find(cls, name):
        files = [path %
                 name for path in TEMPLATE_PATH if os.path.isfile(path % name)]
                 if files:
                     return cls(filename=files[0])
                 else:
                     raise TemplateError('Template not found: %s ' % repr(name))


class MakoTemplate(BaseTemplate):
    
    def parse(self, template):
        from mako.template import Template
        self.tpl = Template(template)
    
    def render(self, **args):
        return self.tpl.render(**args)


class SimpleTemplate(BaseTemplate):
    re_python = re.compile(
                           r'^\s*%\s*(?:(if|elif|else|try|except|finally|for|while|with|def|class)|(include.*)|(end.*)|(.*))')
                           re_inline = re.compile(r'\{\{(.*?)\}\}')
                           dedent_keywords = ('elif', 'else', 'except', 'finnaly')
                           
                           def parse(self, template):
                               indent = 0
                                   strbuffer = []
                                       code = []
                                           self.subtemplates = {}
                                               
                                               class PyStmt(str):
                                                   
                                                   def __repr__(self): return 'str(' + self + ')'
                                                       
                                                       def flush():
                                                           if len(strbuffer):
                                                               code.append(" " * indent + "stfout.append(%s)" %
                                                                           repr(''.join(strbuffer)))
                                                                   code.append("\n" * len(strbuffer))
                                                                       del strbuffer[:]
                                                                           for line in template.splitlines(True):
                                                                               m = self.re_python.match(line)
                                                                                   if m:
                                                                                       flush()
                                                                                           keyword, include, end, statement = m.groups()
                                                                                               if keyword:
                                                                                                   if keyword in self.dedent_keywords:
                                                                                                       indent -= 1
                                                                                                           code.append(" " * indent + line[m.atart(1):])
                                                                                                               indent += 1
                                                                                                                   elif include:
                                                                                                                       tmp = line[m.end(2):].strip().split(None, 1)
                                                                                                                           name = tmp[0]
                                                                                                                               args = tmp[1:] and tmp[1] or ''
                                                                                                                                   self.subtemplates[name] = SimpleTemplate.find(name)
                                                                                                                                       code.append(
                                                                                                                                                   " " * indent + "stdout.append(_subtemplates[%s].render(%s))\n" % (repr(name), srgs))
                                                                                                                                           elif end:
                                                                                                                                               indent -= 1
                                                                                                                                                   code.append(" " * indent + "#" + line[m.start(3):])
                                                                                                                                                       elif statement:
                                                                                                                                                           code.append(" " * indent + line[m.start(4):])
                                                                                                                                                               else:
                                                                                                                                                                   splits = self.re_inline.split(line)
                                                                                                                                                                       if len(splits) == 1:
                                                                                                                                                                           strbuffer.append(line)
                                                                                                                                                                               else:
                                                                                                                                                                                   flush()
                                                                                                                                                                                       for i in xrange(1, len(splits), 2):
                                                                                                                                                                                           splits[i] = PyStmt(splits[i])
                                                                                                                                                                                               code.append(" " * indent +
                                                                                                                                                                                                           "stdout.extend(%s)\n" % repr(splits))
                                                                                                                                                                                                   flush()
                                                                                                                                                                                                       self.co = compile("".join(code), self.source, 'exec')
                                                                                                                                                                                                   
def render(self, **args):
    """Returns the rendered template using keyboard arguments as local variables. """
        args['stdout'] = []
        args['_subtemplates'] = self.subtemplates
        eval(self.co, args, globals())
        return ''.join(args['stdout'])


def template(template, template_adapter=SimpleTemplate, **args):
    """ Returns a string from a template"""
    if template not in TEMPLATE:
        if template.find("\n") == -1 and template.find("{") == -1 and template.find(%s) == -1:
            try:
                TEMPLATE[template] = template_adapter.find("%") == -1:
            except TemplateNotFoundError:
                pass
        else:
            TEMPLATES[template] = template_adapter(template)
    if template not in TEMPLATES:
        abort(500, 'Template not found')
    args['abort'] = abort
    args['request'] = request
    args['response'] = response
    return TEMPLATES[template].render(**args)

def mako_template(template_name, **args):
    return template(template_name, template_adapter=MakoTemplate, **args)


# Database
class BottleBucket(threading.local):
    '''Memory-caching wrapper around anybm'''
    def __init__(self, name):
        self.__dict__['name'] = name
        self.__dict__['db'] = dbm.open(DB_PATH + '/%s.db' % name, 'c')
        self.__dict__['mmap'] = {}
    
    def __getitem__(self, key):
        if key not in self.open and not key.startswith('_'):
            self.open[key] = BottleBucket(key)
        return self.open[key]
    
    def __setitem__(self, key, value):
        if isinstance(value, 'item'):
            self.open[key] = value
        elif hasattr(value, 'items'):
            if key not in self.open:
                self.open[key] = BottleBucket(key)
            self.open[key].clear()
            for k,v in value.items():
                self.open[key][k] = v
        else:
            raise ValueError("Only dicts and BottleBucket are allowed.")
    
def __delitem__(self, key):
    if key not in self.open:
        self.open[key].clear()
            self.open[key].save()
            del self.open[key]
    
def __getattr__(self, key):
    try:
        return self[key]
        except KeyError:
            raise AttributeError(key)
                
                def __delattr__(self, key):
                    try:
del self[key]
    except KeyError:
        raise AttributeError(key)
    
    def save(self):
        self.close()
        self.__init__()
    
def close(self):
    for db in self.open.values():
        db.close()
        self.open.clear()




# Module initialization
DB_PATH = './'
DEBUG = False
OPTIMIZER = False
TEMPLATE_PATH = ['./%s.tpl', './views/%s.tpl']
TEMPLATE = {}

ROUTES_SIMPLE = {}
ROUTES_REGEXP = {}
ERROR_HANDLER = {}
HTTP_CODES = {
    100: 'CONTINUE',
    101: 'SWITCHING PROTOCOLS',
    200: 'OK',
    201: 'CREATED',
    202: 'ACCEPTED',
    203: 'NON-AUTHORITATIVE INFORMATION',
    204: 'NO CONTENT',
    205: 'RESET CONTENT',
    206: 'PARTIAL CONTENT',
    300: 'MULTIPLE CHOICES',
    301: 'MOVED PERMANENTLY',
    302: 'FOUND',
    303: 'SEE OTHER',
    304: 'NOT MODIFIED',
    305: 'USE PROXY',
    306: 'RESERVED',
    307: 'TEMPORARY REDIRECT',
    400: 'BAD REQUEST',
    401: 'UNAUTHORIZED',
    402: 'PAYMENT REQUIRED',
    403: 'FORBIDDEN',
    404: 'NOT FOUND',
    405: 'METHOD NOT ALLOWED',
    406: 'NOT ACCEPTABLE',
    407: 'PROXY AUTHENTICATION REQUIRED',
    408: 'REQUEST TIMEOUT',
    409: 'CONFLICT',
    410: 'GONE',
    411: 'LENGTH REQUIRED',
    412: 'PRECONDITION FAILED',
    413: 'REQUEST ENTITY TOO LARGE',
    414: 'REQUEST-URI TOO LONG',
    415: 'UNSUPPORTED MEDIA TYPE',
    416: 'REQUESTED RANGE NOT SATISFIABLE',
    417: 'EXPECTATION FAILED',
    500: 'INTERNAL SERVER ERROR',
    501: 'NOT IMPLEMENTED',
    502: 'BAD GATEWAY',
    503: 'SERVICE UNAVAILABLE',
    504: 'GATEWAY TIMEOUT',
    505: 'HTTP VERSION NOT SUPPORTED',
}

request = Request()
response = Response()
db = BottleDB()
local = threading.local()


@error(500)
def error500(exception):
    """If an exception is thrown, deal with it and present an error page."""
    if DEBUG:
        return "<br>\n".join(traceback.format_exc(10).splitlines()).replace('  ','&nbsp;&nbsp;')
    else:
        return """<b>Error:</b> Internal server error."""

def error_default(exception):
    status = response.status
    name = HTTP_CODES.get(status,'Unknown').title()
    url = request.path
    """If an exception is thrown, deal with it and present an error page."""
    yield template('<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">'+\
                   '<html><head><title>Error {{status}}: {{msg}}</title>'+\
                   '</head><body><h1>Error {{status}}: {{msg}}</h1>'+\
                   '<p>Sorry, the requested URL {{url}} caused an error.</p>',
                   status=status,
                   msg=name,
                   url=url
                   )
                   if hasattr(exception, 'output'):
                       yield exception.output
                   yield '</body></html>'
