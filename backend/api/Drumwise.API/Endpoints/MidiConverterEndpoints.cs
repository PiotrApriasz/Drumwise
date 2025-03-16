using Drumwise.Application.Common.Extensions;
using Drumwise.Features.AutoDrummingEvaluator;
using Drumwise.Features.MidiConverter;
using MediatR;
using Microsoft.AspNetCore.Mvc;

namespace Drumwise.API.Endpoints;

public static class MidiConverterEndpoints
{
    public static void MapMidiConverterEndpoints(this IEndpointRouteBuilder endpoints)
    {
        var drumsEvaluatingGroup = endpoints.MapGroup(ApiPaths.MidiConverterRootApi).RequireAuthorization();

        drumsEvaluatingGroup.MapPost(ApiPaths.InitiateAudioConverting, async Task<IResult>
                ([FromForm] InitiateConversionCommand uploadDrumsAudioCommand, [FromServices] ISender sender) =>
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