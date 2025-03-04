d3.select("#buttons").datum({ portion: 0 });
// the chain select here pushes the datum onto the up and down buttons also
d3.select("#buttons")
  .select("#previous")
  .on("click", function(d) {
    d.portion -= 20;
    redraw(d.portion);
  });
d3.select("#buttons")
  .select("#next")
  .on("click", function(d) {
    d.portion += 20;
    redraw(d.portion);
  });

function redraw(start) {
  d3.select("table")
    .selectAll("tr")
    .style("display", function(d, i) {
      return i >= start && i < start + 20 ? null : "none";
    });
}

d3.csv(
  "https://gist.githubusercontent.com/jlee-snn/761e57be4a87373f99c49eb420dc4420/raw/e351fd0ccc5f7698230e0955fe44493047d120b8/test_with_scores.csv",
  function(error, data) {
    if (error) {
      console.log("There's an error");
    }
    // We'll be using simpler data as values, not objects.
    var temp_array = [];
    data.forEach(function(d, i) {
      // now we add another data object value, a calculated value.
      // here we are making strings into numbers using type coercion
      d.index = +d.index;
      d.Text = d.Text;
      d.atheism = +d.atheism;
      d.christian = +d.christian;

      // Add a new array with the values of each:
      temp_array.push([
        d.index,
        d.Text,
        d3.format(".1%")(d.atheism),
        d3.format(".1%")(d.christian)
      ]);
    });
    console.log(data);
    console.log(temp_array);
    // create table
    var table = d3.select("#emailist").append("table");
    // create table header
    var columns = ["Index", "Text", "Atheism", "Christian"];
    table
      .append("thead")
      .append("tr")
      .selectAll("th")
      .data(columns)
      .enter()
      .append("th")
      .text(function(d) {
        return d;
      });
    // create table body
    table
      .append("tbody")
      .selectAll("tr")
      .data(temp_array)
      .enter()
      .append("tr")
      .selectAll("td")
      .data(function(d) {
        console.log(d);
        return d;
      })
      .enter()
      .append("td")
      .text(function(d) {
        return d;
      })
      .style("max-width", "800px")
      .style("overflow", "hidden")
      .style("text-overflow", "ellipsis")
      .style("white-space", "nowrap");
    // if we want to show more, enable the code below
    // .style("white-space", "wrap")
    redraw(0);
  }
);

function wrapText(context, text, x, y, maxWidth, lineHeight) {
  var words = text.split(" ");
  var line = "";

  for (var n = 0; n < words.length; n++) {
    var testLine = line + words[n] + " ";
    var metrics = context.measureText(testLine);
    var testWidth = metrics.width;
    if (testWidth > maxWidth && n > 0) {
      context.fillText(line, x, y);
      line = words[n] + " ";
      y += lineHeight;
    } else {
      line = testLine;
    }
  }
  context.fillText(line, x, y);
}

d3.csv(
  "https://raw.githubusercontent.com/jlee-snn/w209_MOSS/master/test_with_scores_example_1.csv",
  function(error, data) {
    if (error) {
      console.log("There's an error");
    }
    var temp_array = [];
    data.forEach(function(d, i) {
      d.index = +d.index;
      d.Text = d.Text;
      temp_array.push([d.index, d.Text]);
    });

    var canvas = document.getElementById("myCanvas");
    var ctx = canvas.getContext("2d");
    var maxWidth = 400;
    var lineHeight = 25;
    var x = (canvas.width - maxWidth) / 2;
    var y = 60;
    // currently hardcoded to the first example.
    var text = temp_array[0][1];
    ctx.font = "15px Arial";
    wrapText(ctx, text, x, y, maxWidth, lineHeight);
  }
);

var margin = { top: 20, right: 30, bottom: 40, left: 30 },
  width = 600 - margin.left - margin.right,
  height = 500 - margin.top - margin.bottom;

var x = d3.scale.linear().range([0, width]);

var y = d3.scale.ordinal().rangeRoundBands([0, height], 0.2);

var xAxis = d3.svg
  .axis()
  .scale(x)
  .orient("bottom");

var yAxis = d3.svg
  .axis()
  .scale(y)
  .orient("left")
  .tickSize(0)
  .tickPadding(6);

var svg = d3
  .select("#barchart")
  .append("svg")
  .attr("width", width + margin.left + margin.right)
  .attr("height", height + margin.top + margin.bottom)
  .append("g")
  .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

d3.csv(
  "https://raw.githubusercontent.com/iatechicken/temp_repo/master/singlecsv.csv",
  type,
  function(error, data) {
    x.domain(
      d3.extent(data, function(d) {
        return d.value;
      })
    ).nice();
    y.domain(
      data.map(function(d) {
        return d.word;
      })
    );

    svg
      .selectAll(".bar")
      .data(data)
      .enter()
      .append("rect")
      .attr("class", function(d) {
        return "bar bar--" + (d.value < 0 ? "negative" : "positive");
      })
      .attr("x", function(d) {
        return x(Math.min(0, d.value));
      })
      .attr("y", function(d) {
        return y(d.word);
      })
      .attr("width", function(d) {
        return Math.abs(x(d.value) - x(0));
      })
      .attr("height", y.rangeBand());

    svg
      .append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis);

    svg
      .append("g")
      .attr("class", "y axis")
      .attr("transform", "translate(" + x(0) + ",0)")
      .call(yAxis);
  }
);

function type(d) {
  d.value = +d.value;
  return d;
}
