using Drumwise.Application.Common.Extensions;
using Drumwise.Features.AutoDrummingEvaluator;
using Drumwise.Features.Homeworks;
using MediatR;
using Microsoft.AspNetCore.Mvc;

namespace Drumwise.API.Endpoints;

public static class DrumsEvaluatingEndpoints
{
    public static void MapDrumsEvaluatingEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var drumsEvaluatingGroup = endpoints.MapGroup("/drumsEvaluating").RequireAuthorization();

        drumsEvaluatingGroup.MapPost("/", async Task<IResult>
                ([FromForm] UploadDrumsAudioCommand uploadDrumsAudioCommand, [FromServices] ISender sender) =>
            {
                var result = await sender.Send(uploadDrumsAudioCommand).ConfigureAwait(false);

                return result.ProduceApiResponse();
            })
            .Produces(StatusCodes.Status200OK)
            .ProducesProblem(StatusCodes.Status400BadRequest)
            .ProducesProblem(StatusCodes.Status500InternalServerError)
            .DisableAntiforgery();
    }
}