/**
 * 旧地址重定向。
 *
 * `source/_data/legacy-redirects.yml` 记录「旧路径 -> 新路径」，
 * 这里为每条记录生成一个静态跳转页，保证旧链接仍然可达。
 * 路径规则变化后（例如再次调整 permalink）无需保留本文件即可删除。
 */
'use strict';

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

hexo.extend.generator.register('legacy_redirects', function () {
  const file = path.join(hexo.base_dir, 'source', '_data', 'legacy-redirects.yml');
  if (!fs.existsSync(file)) return [];

  const map = yaml.load(fs.readFileSync(file, 'utf8')) || {};
  const siteUrl = String(hexo.config.url || '').replace(/\/+$/, '');

  return Object.keys(map).map(function (from) {
    const to = map[from];
    const target = /^https?:\/\//.test(to) ? to : siteUrl + to;
    const safe = escapeHtml(target);
    const html = [
      '<!DOCTYPE html>',
      '<html lang="zh-CN">',
      '<head>',
      '<meta charset="utf-8">',
      '<meta name="robots" content="noindex,follow">',
      '<title>页面已迁移</title>',
      '<link rel="canonical" href="' + safe + '">',
      '<meta http-equiv="refresh" content="0; url=' + safe + '">',
      '<script>location.replace(' + JSON.stringify(target).replace(/</g, '\\u003c') + ')</script>',
      '</head>',
      '<body>',
      '<p>此页面已迁移至 <a href="' + safe + '">' + safe + '</a>。</p>',
      '</body>',
      '</html>',
      '',
    ].join('\n');

    return {
      path: from.replace(/^\/+/, '').replace(/\/+$/, '') + '/index.html',
      data: html,
    };
  });
});
