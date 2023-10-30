let prev_selected = null;

function make_selected(el) {
    if (player.paused) {
        if (prev_selected) {
            prev_selected.classList.remove('selected');
        }
        el.classList.add('selected');
        prev_selected = el;
    }
}

const player = document.getElementById('player');
let pause_after = -1;

setInterval(() => {
    if (pause_after > 0 && player.currentTime > pause_after) {
        player.pause();
        pause_after = -1;
    }
}, 100);

function play(from, to) {
    player.currentTime = from;
    pause_after = to;
    player.play();
}

const segments = Array.from(document.querySelectorAll('.seg--data')).filter(el => el.getAttribute('data-start') !== '');

function scroll_into_view() {
    const time = player.currentTime;
    const seg = segments.find(el => el.getAttribute('data-start') <= time && el.getAttribute('data-end') > time);
    if (seg) {
        seg.scrollIntoView();
    }
}

function highlight_words() {
    const time = player.currentTime;
    const seg = segments.find(el => el.getAttribute('data-start') <= time && el.getAttribute('data-end') > time);
    if (seg) {
        seg.classList.add('selected');
        prev_selected = seg;

        const recoword = Array.from(seg.querySelectorAll('reco-words word')).find(el => el.getAttribute('t-s') <= time && el.getAttribute('t-e') > time);
        if (recoword) {
            let i = recoword.getAttribute('i');
            let reco = seg.getElementsByClassName('reco')[0];
            let words = reco.innerText.split(' ');
            words[i] = '<span class="wordsel">' + words[i] + '</span>';
            reco.innerHTML = words.join(' ');
        }

        const word = Array.from(seg.querySelectorAll('words word')).find(el => el.getAttribute('t-s') <= time && el.getAttribute('t-e') > time);
        if (word) {
            let i = word.getAttribute('i');
            let cs = word.getAttribute('c-s');
            let ce = word.getAttribute('c-e');
            let norm = seg.getElementsByClassName('norm')[0];
            let words = norm.innerText.split(' ');
            words[i] = '<span class="wordsel">' + words[i] + '</span>';
            seg.getElementsByClassName('norm')[0].innerHTML = words.join(' ');
            let text = seg.getElementsByClassName('text')[0];
            words = text.innerText;
            text.innerHTML = words.substring(0, cs) + '<span class="wordsel">' + words.substring(cs, ce) + '</span>' + words.substring(ce);
        }
    }
}

let last_time = player.currentTime;
setInterval(() => {
    if (player.currentTime !== last_time) {
        if (prev_selected) {
            prev_selected.classList.remove('selected');
        }
        last_time = player.currentTime;
        highlight_words();
    }
}, 300);