// ==UserScript==
// @name         Github Issue TItle Recommender
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  iTiger Userscript
// @author       You
// @match        *github.com/*/*/issues/new*
// @icon         https://www.google.com/s2/favicons?sz=64&domain=hibbard.eu
// @grant       GM_xmlhttpRequest
// ==/UserScript==

(function(){
    'use strict';
    console.log("running");
    addButton('Get Title Suggestion', setTitle);
    function addButton(text, onclick, cssObj) {
        let headerActionElement = document.getElementsByClassName('tabnav-tabs')[0];
        console.log('headerActionElement', headerActionElement);

        cssObj = cssObj || {};
        let button = document.createElement('button'), btnStyle = button.style;
        headerActionElement.appendChild(button);
        button.innerHTML = text;
        button.onclick = onclick;
        button.classList = "btn btn-sm flex-md-auto";
        button.type = "button";
        Object.keys(cssObj).forEach(key => btnStyle[key] = cssObj[key]);
        return button;
    }

    function findElementsBySelectorAndText(selector, text) {
        var elements = document.querySelectorAll(selector);
        return Array.prototype.filter.call(elements, function(element){
            return RegExp(text).test(element.textContent);
        });
    }

    function setTitle() {
         var desc = document.getElementById("issue_body").value;
         var base_url = "" //fill with iTiger's backend endpoint
         var path = "predict?text="
         var full_url = base_url + "/" + path + desc
         console.log(full_url)
         GM_xmlhttpRequest({
             method: "GET",
             url: full_url,
             headers: {},
             onload: function (response) {
                 var title;
                 title = JSON.parse(response.response).title;
                 var titleElement;
                 titleElement = document.getElementById("issue_title");
                 titleElement.value = title
             }
         });
        }

}());