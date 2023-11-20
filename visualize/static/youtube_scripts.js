let tag = document.createElement('script');

tag.src = "https://www.youtube.com/iframe_api";
let firstScriptTag = document.getElementsByTagName('script')[0];
firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);


let player;

function onYouTubeIframeAPIReady() {
    player = new YT.Player('player', {
        height: '220',
        width: '360',
        videoId: youtube_video_id,
        playerVars: {
            'playsinline': 1
        },
        events: {}
    });
}

let prev_selected = null;

function make_selected(el) {
    if (el.classList.contains('selected')) {
        el.classList.remove('selected');
        prev_selected = null;
    } else {
        if (player.getPlayerState() != 1 && el !== prev_selected) {
            if (prev_selected) {
                prev_selected.classList.remove('selected');
            }
            el.classList.add('selected');
            el.scrollIntoView({block: 'center'});
            prev_selected = el;
        }
    }
}

let pause_after = -1;

setInterval(() => {
    if (pause_after > 0 && player.getCurrentTime() > pause_after) {
        player.pauseVideo();
        pause_after = -1;
    }
}, 100);

function play(from, to) {
    player.seekTo(from);
    pause_after = to;
    player.playVideo();
}

function stepback() {
    player.seekTo(player.getCurrentTime() - 1.0);
}

function stepforward() {
    player.seekTo(player.getCurrentTime() + 1.0);
}

const segments = Array.from(document.querySelectorAll('.seg--data')).filter(el => el.getAttribute('data-start') !== '');

function scroll_into_view() {
    const time = player.getCurrentTime();
    const seg = segments.find(el => el.getAttribute('data-start') <= time && el.getAttribute('data-end') > time);
    if (seg) {
        seg.scrollIntoView();
    }
}

function highlight_words() {
    const time = player.getCurrentTime();
    const seg = segments.find(el => el.getAttribute('data-start') <= time && el.getAttribute('data-end') > time);
    if (seg) {
        if (seg !== prev_selected) {
            if (prev_selected) {
                for (let sub of ['norm', 'reco', 'text']) {
                    let el = prev_selected.getElementsByClassName(sub);
                    if (el.length > 0) {
                        el[0].innerHTML = el[0].innerText;
                    }
                }
                prev_selected.classList.remove('selected');
            }
            seg.classList.add('selected');
            prev_selected = seg;
        }

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

let last_time = -1;
setInterval(() => {
    if (player.getCurrentTime() !== last_time) {
        last_time = player.getCurrentTime();
        highlight_words();
    }
}, 300);