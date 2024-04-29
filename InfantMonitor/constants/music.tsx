export type MusicPopularDataType = {
    state: "popular",
    sheetURL: string,
    musescoreURL: string,
    title: string,
    sampleImg: string,
    shareURL: string,
    views: string,
    likes: string,
    stars: number,
}

export type MusicNormalDataData = {
    state: "normal",
    sheetURL: string,
    shareURL: string,
    musescoreURL: string,
    title: string,
    sampleImg: string,
}


export type MusicDataType = MusicPopularDataType | MusicNormalDataData

export const musicData: MusicDataType[] = [
    {
        "state": "popular",
        "sheetURL": "https://musescore.com/user/11455091/scores/5904894/embed",
        "musescoreURL": "https://musescore.com/user/11455091/scores/5904894/s/3eN2z1",
        "shareURL": "https://musescore.com/user/11455091/scores/5904894?share=copy_link",
        "title": "Yesterday イエスタデイ - Official髭男dism",
        "sampleImg": "/static/portfolio/music/yesterday.webp",
        "views": "7k",
        "likes": "22",
        "stars": 4.9,
    },
    {
        "state": "normal",
        "sheetURL": "https://musescore.com/user/11455091/scores/5826228/embed",
        "musescoreURL": "https://musescore.com/user/11455091/scores/5826228/s/JTYsva",
        "shareURL": "https://musescore.com/user/11455091/scores/5826228?share=copy_link",
        "title": "Phoenix - Cailin Russo and Chrissy Costanza",
        "sampleImg": "/static/portfolio/music/phoenix.webp",
    },
    {
        "state": "normal",
        "sheetURL": "https://musescore.com/user/11455091/scores/6307844/embed",
        "shareURL": "https://musescore.com/user/11455091/scores/6307844?share=copy_link",
        "musescoreURL": "https://musescore.com/user/11455091/scores/6307844/s/nIJ4rS",
        "title": "Spring Song 春はゆく - Aimer",
        "sampleImg": "/static/portfolio/music/spring.webp",
    },
    {
        "state": "normal",
        "sheetURL": "https://musescore.com/user/11455091/scores/6253349/embed",
        "shareURL": "https://musescore.com/user/11455091/scores/6253349?share=copy_link",
        "musescoreURL": "https://musescore.com/user/11455091/scores/6253349/s/gJ9Z9O",
        "title": "Yuki No Hate Ni Kimi No Na Wo 雪の果てに君の名を - Nonoc",
        "sampleImg": "/static/portfolio/music/re0.webp",
    }
]