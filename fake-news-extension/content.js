function extractArticleText(){

    let article = document.querySelector("article");

    if(article){
        return article.innerText.substring(0,2000);
    }

    let paragraphs = document.querySelectorAll("p");

    let text = "";

    paragraphs.forEach(p=>{
        text += p.innerText + " ";
    });

    return text.substring(0,2000);
}


function showBanner(result, confidence){

    let banner = document.createElement("div");

    banner.style.position = "fixed";
    banner.style.top = "0";
    banner.style.left = "0";
    banner.style.width = "100%";
    banner.style.padding = "15px";
    banner.style.fontSize = "18px";
    banner.style.fontWeight = "bold";
    banner.style.zIndex = "999999";
    banner.style.textAlign = "center";

    if(result === "Fake News"){
        banner.style.backgroundColor = "#ff4444";
        banner.innerText = "⚠️ Potential Fake News Detected (Confidence: " + confidence + ")";
    }else{
        banner.style.backgroundColor = "#4CAF50";
        banner.innerText = "✔ News appears credible (Confidence: " + confidence + ")";
    }

    document.body.appendChild(banner);
}


function highlightArticle(){

    let article = document.querySelector("article");

    if(article){
        article.style.border = "6px solid red";
        article.style.padding = "10px";
    }

}


async function getImageBase64(){

    let img = document.querySelector("img");

    if(!img){
        return "";
    }

    try{

        let res = await fetch(img.src);

        let blob = await res.blob();

        return await new Promise(resolve => {

            let reader = new FileReader();

            reader.onloadend = function(){

                resolve(reader.result.split(",")[1]);

            };

            reader.readAsDataURL(blob);

        });

    }catch(e){

        return "";

    }

}


async function analyzeArticle(){

    let articleText = extractArticleText();

    let imageBase64 = await getImageBase64();

    let response = await fetch("http://127.0.0.1:8000/predict",{

        method:"POST",

        headers:{
            "Content-Type":"application/json"
        },

        body:JSON.stringify({

            text: articleText,

            image: imageBase64

        })

    });

    let data = await response.json();

    showBanner(data.prediction, data.confidence);

    if(data.prediction === "Fake News"){

        highlightArticle();

    }

}


window.addEventListener("load", () => {

    setTimeout(()=>{

        analyzeArticle();

    },2000);

});