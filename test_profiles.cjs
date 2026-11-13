// Run with: node --test test_profiles.cjs
const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const source = fs.readFileSync(`${__dirname}/app.py`, 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const PROFILES = 'llm_checker_endpoints_v1';
const CONFIG = 'llm_checker_config_v1';

function page(storage = new Map()) {
  const elements = new Map();
  const callbacks = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, {
          value: '', textContent: '', className: '', disabled: false,
          defaultValue: id === 'base_url' ? 'https://api.openai.com/v1' : '',
          options: [], selectedIndex: 0,
          addEventListener() {},
          replaceChildren(...options) { this.options = options; this.value = options[0].value; },
          add(option) { this.options.push(option); },
        });
      }
      return elements.get(id);
    },
    addEventListener(event, callback) { callbacks.set(event, callback); },
    querySelectorAll() { return []; },
  };
  const localStorage = {
    getItem(key) { return storage.get(key) ?? null; },
    setItem(key, value) { storage.set(key, value); },
  };
  const context = vm.createContext({
    document, localStorage, URL,
    Option: function(text, value) { this.text = text; this.value = value; },
  });
  vm.runInContext(source, context);
  callbacks.get('DOMContentLoaded')();
  return {
    storage, context, localStorage,
    element: id => document.getElementById(id),
    run: expression => vm.runInContext(expression, context),
    profiles: () => JSON.parse(storage.get(PROFILES) || '[]'),
    fill(name, url, key) {
      document.getElementById('profile_name').value = name;
      document.getElementById('base_url').value = url;
      document.getElementById('api_key').value = key;
    },
  };
}

test('separate profiles restore the correct URL and key, including after reopening', () => {
  let app = page();
  app.fill('Provider A', 'https://a.example/v1/', 'key-a');
  app.run('saveEndpoint()');
  const firstId = app.profiles()[0].id;
  app.run('newEndpoint()');
  assert.equal(app.element('api_key').value, '');
  app.fill('Provider B', 'https://b.example/v1', 'key-b');
  app.run('saveEndpoint()');
  assert.equal(app.profiles().length, 2);
  app.element('endpoint_profile').value = firstId;
  app.element('timeout').value = '45';
  app.run('selectEndpoint()');
  assert.equal(app.element('base_url').value, 'https://a.example/v1');
  assert.equal(app.element('api_key').value, 'key-a');
  assert.equal(app.element('timeout').value, '45');
  app = page(app.storage);
  assert.equal(app.element('endpoint_profile').value, firstId);
  assert.equal(app.element('base_url').value, 'https://a.example/v1');
  assert.equal(app.element('api_key').value, 'key-a');
  assert.equal(app.element('profile_name').value, 'Provider A');
  assert.equal(app.element('delete_endpoint').disabled, false);
});

test('updating replaces the selected profile and deleting clears its active credentials', () => {
  const app = page();
  app.fill('A', 'https://same.example/v1', 'key-a');
  app.run('saveEndpoint()');
  app.run('newEndpoint()');
  app.fill('B', 'https://same.example/v1', 'key-b');
  app.run('saveEndpoint()');
  const secondId = app.profiles()[1].id;
  app.fill('B renamed', 'https://new.example/v1', 'key-new');
  app.run('saveEndpoint()');
  assert.equal(app.profiles().length, 2);
  assert.deepEqual(app.profiles()[1], {
    id: secondId, name: 'B renamed', base_url: 'https://new.example/v1', api_key: 'key-new',
  });
  app.run('deleteEndpoint()');
  assert.equal(app.profiles().length, 1);
  assert.equal(app.profiles()[0].api_key, 'key-a');
  assert.equal(app.element('api_key').value, '');
  assert.equal(app.element('endpoint_profile').value, '');
  assert.equal(app.element('delete_endpoint').disabled, true);
  assert.equal(JSON.parse(app.storage.get(CONFIG)).api_key, '');
  const reopened = page(app.storage);
  assert.equal(reopened.element('api_key').value, '');
});

test('existing config remains usable without profiles and missing names use the hostname', () => {
  const app = page(new Map([[CONFIG, JSON.stringify({base_url: 'https://old.example/v1', api_key: 'old-key'})]]));
  assert.equal(app.element('api_key').value, 'old-key');
  app.run('saveEndpoint()');
  assert.equal(app.profiles()[0].name, 'old.example');
  assert.equal(app.profiles()[0].api_key, 'old-key');
});

test('invalid URLs and empty keys are rejected without changing saved profiles', () => {
  const app = page();
  for (const url of ['not-a-url', 'javascript:alert(1)', 'https://user:secret@host.example']) {
    app.fill('Invalid', url, 'key');
    app.run('saveEndpoint()');
    assert.equal(app.profiles().length, 0);
    assert.match(app.element('profile_status').className, /error/);
  }
  app.fill('No key', 'https://valid.example/v1', '  ');
  app.run('saveEndpoint()');
  assert.equal(app.profiles().length, 0);
  assert.match(app.element('profile_status').textContent, /API key/);
});

test('corrupt profile storage does not break initialization', () => {
  for (const value of ['{broken', '{}', '[null, {}, {"id":1}]']) {
    const app = page(new Map([[PROFILES, value]]));
    assert.equal(app.element('endpoint_profile').options.length, 1);
    app.fill('Valid', 'https://valid.example/v1', 'key');
    app.run('saveEndpoint()');
    assert.equal(app.profiles().length, 1);
  }
});

test('storage failures report an error and do not pretend to save or delete', () => {
  const app = page();
  app.fill('Saved', 'https://saved.example/v1', 'key');
  app.run('saveEndpoint()');
  const original = app.storage.get(PROFILES);
  app.localStorage.setItem = () => { throw new Error('Storage unavailable'); };
  app.fill('Changed', 'https://changed.example/v1', 'changed-key');
  app.run('saveEndpoint()');
  assert.equal(app.storage.get(PROFILES), original);
  assert.match(app.element('profile_status').className, /error/);
  app.run('deleteEndpoint()');
  assert.equal(app.storage.get(PROFILES), original);
  assert.equal(app.element('api_key').value, 'changed-key');
});
